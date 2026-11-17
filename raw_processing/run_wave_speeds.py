#!/usr/bin/env python3
"""Linear and conjugate-flow wave speeds for prescribed stratification.

Dependencies: NumPy and SciPy; Matplotlib only for calc_c0(plot_phi=True).

Functions:
    calc_c0(z, N2, dz) -> (c0, alpha_kdv)
    no_core_djl_solver(N2, z, ...) -> result dictionary
    smooth_core_djl_solver(N2, z, ...) -> result dictionary

For the nonlinear solvers, use a uniform z grid including 0 and 1 and
nonnegative dimensionless N2 = N2_dim * H / gprime. Speeds are then in units
of sqrt(gprime * H), and displacements are in units of H. calc_c0 instead
uses the units supplied by its inputs. Its alpha is the KdV nonlinearity
coefficient, distinct from the initial shooting slope in the nonlinear solvers.

Example (with arrays z and N2 already defined):
    from wave_speeds import calc_c0, no_core_djl_solver, smooth_core_djl_solver
    c0, alpha_kdv = calc_c0(z, N2, z[1] - z[0])
    no_core = no_core_djl_solver(N2, z, verbose=True)
    core = smooth_core_djl_solver(N2, z, verbose=True)

Each nonlinear branch can raise RuntimeError if its search finds no accepted
solution. See the individual docstrings for bounds, guesses and tolerances.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

__all__ = ['calc_c0', 'no_core_djl_solver', 'smooth_core_djl_solver']


def calc_c0(Z, N2_vals, dz, plot_phi=False):
    """Linear long-wave eigenvalue solver. Returns (c0, alpha).
alpha is the KdV nonlinearity coefficient computed from the linear structure
function, not a DJL shooting slope. With dimensional inputs alpha has units 1/s.
Input units determine speed units; assumes uniform vertical spacing."""
    z_in = Z[1:-1]
    N2_in = N2_vals[1:-1]
    ni = len(z_in)
    main = -2.0 * np.ones(ni) / dz ** 2
    off = np.ones(ni - 1) / dz ** 2
    D2 = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    C = (1.0 / (N2_in + 1e-08))[:, None] * D2
    (eigvals, eigvecs) = np.linalg.eig(C)
    eigvals = np.real_if_close(eigvals)
    eigvecs = np.real_if_close(eigvecs)
    lambda_vals = -eigvals
    pos = lambda_vals > 0
    lambda_pos = lambda_vals[pos]
    phi_pos = eigvecs[:, pos]
    c_vals = 1.0 / np.sqrt(lambda_pos)
    idx = np.argsort(c_vals)[::-1]
    c_sorted = c_vals[idx]
    phi_sorted = phi_pos[:, idx]
    alpha = 1.5 * c_sorted[0] * np.sum(np.diff(abs(phi_sorted[:, 0])) ** 3 / dz ** 2) / np.sum(np.diff(phi_sorted[:, 0]) ** 2 / dz)
    if plot_phi:
        import matplotlib.pyplot as plt
        plt.plot(abs(phi_sorted[:, 0]), Z[1:-1])
    return (c_sorted[0], alpha)


def no_core_djl_solver(
        N2_profile, z, verbose=False, *, initial_guess=None,
        alpha_bounds=(-2.0, -0.05), c_bounds=None,
        residual_tol=1e-6, geometry_tol=1e-6, slope_margin=1e-6,
        amplitude_tol=1e-4, seed_grid_size=5, max_starts=3, max_nfev=100,
        max_step=0.02, rtol=1e-8, atol=1e-10):
    """Solve the no-core branch for (alpha,c), integrating over [0,1].

    Same dimensionless inputs as smooth_core_djl_solver. Use bounded least
    squares to solve eta(1)=0 and I(1)=0, where I'=eta'^3 and eta'(0)=alpha.
    No eta'=1 matching condition is imposed. Accept only when both residuals
    meet residual_tol, max(eta') <= 1-slope_margin, max(abs(eta)) exceeds
    amplitude_tol, and 0 <= z-eta <= 1 within geometry_tol. alpha_bounds
    exclude zero to avoid convergence to the identically zero solution.
    The optimizer scales the two residuals by abs(alpha) and abs(alpha)**3
    to avoid preferring arbitrarily weak shots; final checks use raw residuals.

    initial_guess=(alpha,c) bypasses coarse seeding. Otherwise rank a small
    seed_grid_size squared grid by endpoint residual norm, refining at most
    max_starts guesses. Default c_bounds=(c0,0.5); bounds are configurable.
    A failed local search does not establish that the branch does not exist.
    np.interp endpoint clamping is a numerical continuation for optimizer
    trials only; accepted solutions must pass the reference-coordinate check.
    Return the same profile/speed fields as smooth_core, with h=1,
    core_thickness=0, eta_z_surface, max_eta_z and no_core_margin=1-max_eta_z.
    """
    z = np.asarray(z, dtype=float)
    N2 = np.asarray(N2_profile, dtype=float)
    if (z.ndim != 1 or N2.shape != z.shape or z.size < 4
            or not np.isfinite(z).all() or not np.isfinite(N2).all()):
        raise ValueError('z and N2_profile must be matching finite 1D arrays of length >= 4')
    dz = np.diff(z)
    if (np.any(dz <= 0) or not np.allclose(dz, dz[0])
            or not np.isclose(z[0], 0) or not np.isclose(z[-1], 1)
            or np.any(N2 < 0) or not np.any(N2 > 0)):
        raise ValueError('Need a uniform z grid from 0 to 1 and nonnegative, nonzero N2')
    c0, _ = calc_c0(z, N2, dz[0])
    bounds = np.asarray([alpha_bounds, (c0, 0.5) if c_bounds is None else c_bounds], dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or bounds[0, 1] >= 0 or bounds[1, 0] <= 0):
        raise ValueError('Need ordered finite bounds with alpha < 0 and c > 0')
    if (not np.isfinite([residual_tol, geometry_tol, slope_margin, amplitude_tol,
                         max_step, rtol, atol]).all()
            or min(residual_tol, geometry_tol, slope_margin, amplitude_tol,
                   max_step, rtol, atol) <= 0):
        raise ValueError('Tolerances, margins and max_step must be finite and positive')
    for name, value in [('seed_grid_size', seed_grid_size), ('max_starts', max_starts),
                        ('max_nfev', max_nfev)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    def log(message):
        if verbose:
            print(message, flush=True)

    integrations = 0
    cache = {}

    def eta_zero(height, y):
        # Slope extrema occur where eta=0 (or N2=0, giving a flat slope).
        return y[0]

    def slope_one(height, y):
        # Reference-coordinate extrema occur where eta'=1.
        return y[2] - 1

    def evaluate(parameters, report=True):
        nonlocal integrations
        key = tuple(parameters)
        if key not in cache:
            alpha, c = parameters

            def system(height, y):
                eta, integral, slope = y
                return [slope, slope**3, -np.interp(height-eta, z, N2)*eta/c**2]

            integrations += 1
            sol = solve_ivp(system, (0, 1), (0, 0, alpha), dense_output=True,
                            events=(eta_zero, slope_one), max_step=max_step, rtol=rtol, atol=atol)
            if not sol.success or not np.isfinite(sol.y).all():
                raise RuntimeError(f'No-core shot failed: {sol.message}')
            residual = sol.y[[0, 1], -1]
            cache[key] = (sol, residual)
            if report:
                log(f'  alpha={alpha:.9g}, c={c:.9g}: '
                    f'boundary={residual[0]:+.3e}, momentum={residual[1]:+.3e}')
        return cache[key]

    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float)
        if (guess.shape != (2,) or not np.isfinite(guess).all()
                or np.any(guess < bounds[:, 0]) or np.any(guess > bounds[:, 1])):
            raise ValueError('initial_guess must be (alpha,c) within the supplied bounds')
        seeds = [(0.0, guess)]
    else:
        seeds = []
        log(f'no_core: generating {seed_grid_size**2} coarse seed shots')
        for alpha in np.linspace(*bounds[0], seed_grid_size):
            for c in np.linspace(*bounds[1], seed_grid_size):
                guess = np.array([alpha, c])
                try:
                    _, residual = evaluate(guess, report=False)
                except RuntimeError as exc:
                    log(f'  Seed skipped: {exc}')
                    continue
                scale = np.array([abs(alpha), abs(alpha)**3])
                seeds.append((float(np.linalg.norm(residual / scale)), guess))
        seeds.sort(key=lambda item: item[0])
    best = np.inf
    rejections = []
    for attempt, (_, guess) in enumerate(seeds[:max_starts], 1):
        log(f'No-core local solve {attempt}: alpha={guess[0]:.9g}, c={guess[1]:.9g}')
        try:
            fit = least_squares(lambda x: evaluate(x)[1] / np.array([abs(x[0]), abs(x[0])**3]), guess,
                                bounds=(bounds[:, 0], bounds[:, 1]),
                                x_scale=np.array([1.0, 0.3]), diff_step=1e-4,
                                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=max_nfev)
            sol, residual = evaluate(fit.x)
        except RuntimeError as exc:
            log(f'  Local solve skipped: {exc}')
            continue
        best = min(best, float(np.max(np.abs(residual))))
        if np.any(np.abs(residual) > residual_tol):
            log(f'  Not accepted: {fit.message}; residuals exceed {residual_tol:g}')
            continue
        z_eta = np.linspace(0, 1, 500)
        eta, integral, eta_z = sol.sol(z_eta)
        check_z = np.unique(np.concatenate([z_eta, sol.t, *sol.t_events]))
        checked_eta, _, checked_slope = sol.sol(check_z)
        reference_z = check_z - checked_eta
        max_eta_z = float(np.max(checked_slope))
        max_eta = float(np.max(np.abs(checked_eta)))
        reason = None
        if max_eta <= amplitude_tol:
            reason = 'near-zero displacement (trivial branch)'
        elif max_eta_z > 1 - slope_margin:
            reason = f'no-core condition failed: max eta_z={max_eta_z:.9g}'
        elif np.min(reference_z) < -geometry_tol or np.max(reference_z) > 1 + geometry_tol:
            reason = 'profile leaves the fluid reference interval [0,1]'
        if reason is not None:
            log(f'  Not accepted: {reason}')
            rejections.append(reason)
            continue
        alpha, c = fit.x
        log(f'Accepted no-core solution: alpha={alpha:.9g}, c={c:.9g}, '
            f'max eta_z={max_eta_z:.9g}; {integrations} integrations')
        return dict(c=float(c), c0=c0, h=1.0, core_thickness=0.0,
                    initial_slope=float(alpha), eta_z_h=float(eta_z[-1]),
                    eta_z_surface=float(eta_z[-1]), max_eta_z=max_eta_z,
                    no_core_margin=1-max_eta_z, boundary_residual=float(residual[0]),
                    momentum_residual=float(residual[1]), momentum_integral=float(integral[-1]),
                    max_abs_eta=max_eta, z_eta=z_eta, eta=eta, eta_z=eta_z,
                    N2=N2.copy(), solver='no_core', integrations=integrations,
                    optimizer_message=fit.message)
    details = '; '.join(dict.fromkeys(rejections))
    raise RuntimeError(f'No accepted no-core solution; best maximum residual={best:.6g}; '
                       f'{integrations} integrations. {details} '
                       'Try another initial_guess or bounds; this does not prove nonexistence.')


def smooth_core_djl_solver(
        N2_profile, z, verbose=False, *, initial_guess=None,
        alpha_bounds=(-2.0, -0.05), c_bounds=None,
        h_bounds=(0.001, 0.999), residual_tol=1e-6, geometry_tol=1e-6,
        seed_grid_size=5, max_starts=3, max_nfev=100,
        max_step=0.02, rtol=1e-8, atol=1e-10):
    """Solve for (alpha, c, h) with a smooth trapped-core boundary.

    Use uniform z in [0,1] and dimensionless N2=N2_dim*H/gprime. Integrate
    directly to trial h, with no terminal core-intersection event, and solve
    (eta(h)-h+1, eta'(h)-1, I(h)-h+1) = 0 by bounded least squares.
    Finite differences use diff_step=1e-4 to resolve changes above IVP noise.
    All three residuals must independently meet residual_tol, regardless of
    the optimizer's stopping status. Final geometry checks reject profiles
    outside 0 <= z-eta <= 1 or with an earlier surface contact.

    initial_guess is an optional (alpha, c, h) tuple. Otherwise a small
    seed_grid_size squared grid supplies seeds: integrate until the first
    minimum of eta-z+1 (eta'=1), rank by the three residuals, and refine at
    most max_starts guesses. Default c_bounds=(c0, 0.5). Bounds exclude the
    trivial alpha=0, h=1 solution and are configurable. Solutions outside
    the supplied bounds will not be found. No uniqueness is assumed.

    np.interp clamps stratification at its endpoints for off-domain optimizer
    trials; this is only a numerical continuation. Accepted profiles must pass
    the reference-coordinate geometry checks. No above-surface continuation
    is appended to the returned displacement profile.
    """
    z = np.asarray(z, dtype=float)
    N2 = np.asarray(N2_profile, dtype=float)
    if (z.ndim != 1 or N2.shape != z.shape or z.size < 4
            or not np.isfinite(z).all() or not np.isfinite(N2).all()):
        raise ValueError('z and N2_profile must be matching finite 1D arrays of length >= 4')
    dz = np.diff(z)
    if (np.any(dz <= 0) or not np.allclose(dz, dz[0])
            or not np.isclose(z[0], 0) or not np.isclose(z[-1], 1)
            or np.any(N2 < 0) or not np.any(N2 > 0)):
        raise ValueError('Need a uniform z grid from 0 to 1 and nonnegative, nonzero N2')
    c0, _ = calc_c0(z, N2, dz[0])
    speed_bounds = (c0, 0.5) if c_bounds is None else c_bounds
    bounds = np.asarray([alpha_bounds, speed_bounds, h_bounds], dtype=float)
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or bounds[0, 1] >= 0 or bounds[1, 0] <= 0
            or bounds[2, 0] <= 0 or bounds[2, 1] >= 1):
        raise ValueError('Need ordered finite bounds: alpha < 0, c > 0, 0 < h < 1')
    if (not np.isfinite([residual_tol, geometry_tol, max_step, rtol, atol]).all()
            or min(residual_tol, geometry_tol, max_step, rtol, atol) <= 0):
        raise ValueError('Tolerances and max_step must be finite and positive')
    for name, value in [('seed_grid_size', seed_grid_size), ('max_starts', max_starts),
                        ('max_nfev', max_nfev)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    def log(message):
        if verbose:
            print(message, flush=True)

    integrations = 0

    def slope_match(height, y):
        return y[2] - 1

    def integrate(parameters, seed=False):
        nonlocal integrations
        alpha, c, h = parameters

        def system(height, y):
            eta, integral, slope = y
            return [slope, slope**3, -np.interp(height-eta, z, N2)*eta/c**2]

        # Seed shots stop at their first gap minimum. Optimization shots
        # always reach their prescribed h; stationary points are recorded.
        slope_match.terminal = seed
        slope_match.direction = 1 if seed else 0
        integrations += 1
        sol = solve_ivp(system, (0, h), (0, 0, alpha), dense_output=True,
                        events=slope_match, max_step=max_step, rtol=rtol, atol=atol)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(f'Smooth-core shot failed: {sol.message}')
        return sol

    def endpoint_residual(sol, h):
        eta, integral, slope = sol.sol(h)
        return np.array([eta-h+1, slope-1, integral-h+1])

    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=float)
        if (guess.shape != (3,) or not np.isfinite(guess).all()
                or np.any(guess < bounds[:, 0]) or np.any(guess > bounds[:, 1])):
            raise ValueError('initial_guess must be (alpha,c,h) within the supplied bounds')
        seeds = [(0.0, guess)]
    else:
        seeds = []
        log(f'smooth_core: generating {seed_grid_size**2} coarse seed shots')
        for alpha in np.linspace(*bounds[0], seed_grid_size):
            for c in np.linspace(*bounds[1], seed_grid_size):
                try:
                    sol = integrate((alpha, c, bounds[2, 1]), seed=True)
                except RuntimeError as exc:
                    log(f'  Seed skipped: {exc}')
                    continue
                if not sol.t_events[0].size:
                    continue
                h = float(sol.t_events[0][0])
                if h < bounds[2, 0]:
                    continue
                guess = np.array([alpha, c, h])
                seeds.append((float(np.linalg.norm(endpoint_residual(sol, h))), guess))
        seeds.sort(key=lambda item: item[0])
    if not seeds:
        raise RuntimeError('No smooth-core starting guesses found; supply initial_guess or adjust bounds')
    best = np.inf
    for attempt, (_, guess) in enumerate(seeds[:max_starts], 1):
        log(f'Local solve {attempt}: alpha={guess[0]:.9g}, c={guess[1]:.9g}, h={guess[2]:.9g}')
        cache = {}

        def evaluate(parameters):
            key = tuple(parameters)
            if key not in cache:
                sol = integrate(parameters)
                residual = endpoint_residual(sol, parameters[2])
                cache[key] = (sol, residual)
                log(f'  alpha={parameters[0]:.9g}, c={parameters[1]:.9g}, '
                    f'h={parameters[2]:.9g}: boundary={residual[0]:+.3e}, '
                    f'slope={residual[1]:+.3e}, momentum={residual[2]:+.3e}')
            return cache[key]

        try:
            fit = least_squares(lambda x: evaluate(x)[1], guess,
                                bounds=(bounds[:, 0], bounds[:, 1]),
                                x_scale=np.array([1.0, 0.3, 1.0]), diff_step=1e-4,
                                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=max_nfev)
            sol, residual = evaluate(fit.x)
        except RuntimeError as exc:
            log(f'  Local solve skipped: {exc}')
            continue
        best = min(best, float(np.max(np.abs(residual))))
        if np.any(np.abs(residual) > residual_tol):
            log(f'  Not accepted: {fit.message}; residuals exceed {residual_tol:g}')
            continue
        alpha, c, h = fit.x
        z_eta = np.linspace(0, h, 500)
        eta, integral, eta_z = sol.sol(z_eta)
        # Extrema of z-eta occur at eta'=1. Include these in addition to the
        # plotting grid so a narrow crossing cannot hide between plot points.
        check_z = np.unique(np.concatenate([z_eta, sol.t_events[0]]))
        reference_z = check_z - sol.sol(check_z)[0]
        earlier = sol.t_events[0][sol.t_events[0] < h - 1e-5]
        if (np.min(reference_z) < -geometry_tol or np.max(reference_z) > 1 + geometry_tol
                or (earlier.size and np.any(1 + sol.sol(earlier)[0] - earlier <= geometry_tol))):
            log('  Not accepted: profile leaves the fluid region or has an earlier surface contact')
            continue
        log(f'Accepted smooth core: alpha={alpha:.9g}, c={c:.9g}, h={h:.9g}; '
            f'{integrations} total integrations')
        return dict(c=float(c), c0=c0, h=float(h), core_thickness=float(1-h),
                    initial_slope=float(alpha), eta_z_h=float(eta_z[-1]),
                    boundary_residual=float(residual[0]), slope_residual=float(residual[1]),
                    momentum_residual=float(residual[2]), momentum_integral=float(integral[-1]),
                    max_abs_eta=float(np.max(np.abs(eta))), z_eta=z_eta, eta=eta,
                    eta_z=eta_z, N2=N2.copy(), solver='smooth_core',
                    integrations=integrations, optimizer_message=fit.message)
    raise RuntimeError(f'No accepted smooth-core solution; best maximum residual={best:.6g}; '
                       f'{integrations} integrations. Try another initial_guess or bounds.')


# ------------------------------- batch runner -------------------------------
import argparse
import json
from pathlib import Path
import time


def run_dataset(input_file, output_file, *, verbose=False, checkpoint_every=50):
    """Run selected exported arrivals."""
    input_file, output_file = Path(input_file), Path(output_file)
    with np.load(input_file, allow_pickle=False) as data:
        inputs = {key: data[key] for key in ('indices', 'z_hat', 'N2_hat', 'H', 'speed_scale')}
        if 'grid_shape' not in data:
            raise ValueError('Export must contain grid_shape; re-export from Supplementary_Data/process.ipynb.')
        grid_shape = data['grid_shape']
    if (grid_shape.shape != (2,) or not np.issubdtype(grid_shape.dtype, np.integer)
            or np.any(grid_shape <= 0)):
        raise ValueError('grid_shape must contain two positive integers')
    shape = tuple(int(x) for x in grid_shape)
    indices = inputs['indices']
    if (indices.ndim != 2 or indices.shape[1] != 2
            or not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError('indices must be an N x 2 integer array')
    n = len(indices)
    if (inputs['z_hat'].ndim != 2 or inputs['N2_hat'].shape != inputs['z_hat'].shape
            or inputs['z_hat'].shape[0] != n or inputs['z_hat'].shape[1] < 4
            or inputs['H'].shape != (n,) or inputs['speed_scale'].shape != (n,)):
        raise ValueError('Input array shapes are inconsistent')
    if np.any(indices < 0) or np.any(indices >= np.asarray(shape)):
        raise ValueError('Arrival indices fall outside grid_shape')
    if len(np.unique(indices, axis=0)) != n:
        raise ValueError('Duplicate arrival indices')
    if not output_file.parent.is_dir():
        raise ValueError(f'Output directory does not exist: {output_file.parent}')
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')

    # All numeric outputs are mooring x bore arrays, with NaN outside valid results.
    stats = {key: np.full(shape, np.nan) for key in
             ('c0', 'alpha', 'cDJL', 'etaDJL', 'max_gradient')}
    solver_used = np.full(shape, '', dtype='U16')
    processed = np.zeros(shape, dtype=bool)
    events = []
    counts = {'no_core': 0, 'smooth_core': 0, 'invalid_input': 0, 'linear_failed': 0, 'djl_failed': 0}
    started = time.monotonic()
    log_path = output_file.with_name(output_file.stem + '_log.json')

    def save():
        # File handle prevents NumPy silently appending another suffix.
        temporary = output_file.with_name(output_file.name + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **stats, grid_shape=np.asarray(shape),
                                solver=solver_used, processed=processed)
        temporary.replace(output_file)
        log_path.write_text(json.dumps({
            'input': str(input_file), 'grid_shape': list(shape), 'selected': n,
            'processed': int(processed.sum()), 'counts': counts,
            'surface_N2_threshold': 0.5, 'residual_tol': 5e-3,
            'elapsed_seconds': time.monotonic()-started, 'events': events,
        }, indent=2) + '\n')

    def solve_one(k, Jidx, i):
        z, N2 = inputs['z_hat'][k], inputs['N2_hat'][k]
        H, speed_scale = inputs['H'][k], inputs['speed_scale'][k]
        dz = np.diff(z)
        if (not np.isfinite(z).all() or not np.isfinite(N2).all()
                or not np.isfinite(H) or not np.isfinite(speed_scale)
                or H <= 0 or speed_scale <= 0 or np.any(dz <= 0)
                or not np.allclose(dz, dz[0]) or not np.isclose(z[0], 0)
                or not np.isclose(z[-1], 1) or np.any(N2 < 0) or not np.any(N2 > 0)):
            counts['invalid_input'] += 1
            events.append({'mooring': int(Jidx), 'bore': int(i), 'reason': 'Invalid exported inputs'})
            return
        try:
            c0_hat, alpha_hat = calc_c0(z, N2, dz[0])
            if not np.isfinite([c0_hat, alpha_hat]).all() or c0_hat <= 0:
                raise ValueError('Invalid linear solver output')
            stats['c0'][Jidx, i] = c0_hat * speed_scale       # m/s
            stats['alpha'][Jidx, i] = alpha_hat * speed_scale / H  # 1/s
        except Exception as exc:
            counts['linear_failed'] += 1
            events.append({'mooring': int(Jidx), 'bore': int(i),
                           'linear_error': f'{type(exc).__name__}: {exc}'})
            return

        no_core = ('no_core', no_core_djl_solver)
        smooth_core = ('smooth_core', smooth_core_djl_solver)
        order = (no_core, smooth_core) if N2[-1] < 0.5 else (smooth_core, no_core)
        errors = {}
        for name, solver in order:
            try:
                result = solver(N2, z, verbose=verbose, residual_tol=5e-3)
                # eta_z_h is the original boundary/surface-gradient output.
                c_hat, eta_hat, gradient = result['c'], result['max_abs_eta'], result['eta_z_h']
                if not np.isfinite([c_hat, eta_hat, gradient]).all():
                    raise ValueError('Non-finite nonlinear solver output')
            except Exception as exc:
                errors[name] = f'{type(exc).__name__}: {exc}'
                continue
            stats['cDJL'][Jidx, i] = c_hat * speed_scale     # m/s
            stats['etaDJL'][Jidx, i] = eta_hat * H           # m
            stats['max_gradient'][Jidx, i] = gradient       # dimensionless
            solver_used[Jidx, i] = name
            counts[name] += 1
            break
        else:
            counts['djl_failed'] += 1
        if errors:
            events.append({'mooring': int(Jidx), 'bore': int(i), 'branch_errors': errors,
                           'solver_used': str(solver_used[Jidx, i])})

    print(f'Processing {n} selected mode-1 arrivals on grid {shape}', flush=True)
    try:
        for k, (Jidx, i) in enumerate(indices):
            solve_one(k, Jidx, i)
            processed[Jidx, i] = True
            print('.', end='', flush=True)
            if (k+1) % checkpoint_every == 0:
                save()
                print(f' {k+1}/{n}, elapsed {time.monotonic()-started:.0f}s', flush=True)
    finally:
        # Also save completed arrivals on a normal Python interruption.
        save()
    print(f'\nSaved {output_file} and {log_path}', flush=True)
    print('Linear successes:', int(np.isfinite(stats['c0']).sum()), counts, flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=(
        'Standalone mode-1 wave-speed batch runner. Requires NumPy and SciPy. '
        'Uses no-core first if surface normalized N2 < 0.5, otherwise smooth-core first. '
        'Tries the other branch only on failure; residual_tol=5e-3. '
        'Saves c0 and cDJL in m/s, etaDJL in m, alpha in 1/s, and dimensionless max_gradient '
        '(eta_z_h: boundary gradient, as in the original workflow). '
        'Invalid/unselected results are NaN; linear results survive nonlinear failures.'))
    parser.add_argument('--input', type=Path, default=Path('djl_inputs.npz'))
    parser.add_argument('--output', type=Path, default=Path('djl_stats.npz'))
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=50)
    args = parser.parse_args()
    run_dataset(args.input, args.output, verbose=args.verbose,
                checkpoint_every=args.checkpoint_every)


if __name__ == '__main__':
    main()
