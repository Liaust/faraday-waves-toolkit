#!/usr/bin/env python3
"""Fit the ray-traced FSSS displacement-to-gradient calibration.

The script simulates how known synthetic surfaces distort the observed dot grid,
then fits the linearized mapping

    [grad_x, grad_y]^T = B @ [dx_obj_m, dy_obj_m]^T

where the displacement is apparent dot motion in physical object-plane units and
the gradient is the dimensionless liquid-surface slope.

Expected inputs:
    inputs/calibration_metadata.yaml
    outputs/camera_calibration/<calibration_id>/camera_intrinsics.json
    outputs/full_fsss/calibration/<calibration_id>/camera_pose_grid.json
    outputs/full_fsss/calibration/<calibration_id>/dot_grid_reference.csv
    outputs/full_fsss/calibration/<calibration_id>/flat_liquid_dots.csv
    outputs/full_fsss/calibration/<calibration_id>/optical_geometry_refined.json

Main output:
    outputs/full_fsss/calibration/<calibration_id>/raytrace_fsss_calibration.json
"""

# this simulates rays through the optical stack for known synthetic surfaces
# then fits the matrix that converts apparent dot displacement
# into liquid-surface gradients


import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import least_squares


# path setup
# path_utils gives repo root and repo-relative path helpers
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from path_utils import PROJECT_ROOT, project_path, relpath  # noqa: E402

DEFAULT_RANDOM_SEED = 7
DEFAULT_MAX_NFEV = 80
DEFAULT_ROOT_TOL_M = 1e-10
DEFAULT_MAX_ACCEPTED_RAY_RESIDUAL_M = 2e-5
DEFAULT_PHASES_PER_SURFACE_MODEL = [0.0, 0.73, 1.91]
DEFAULT_CENTRAL_FRACTION_OF_DOTS = 0.80
USE_ONLY_METADATA_SURFACE_MODELS = False




# keep the cli narrow
# metadata finds the earlier calibration files, output-dir picks where json goes
# max-dots limits runtime
def parse_args() -> argparse.Namespace:
    # public cli only exposes metadata, output directory, and max dots
    # solver tolerances and surface-model phases stay fixed inside the script
    parser = argparse.ArgumentParser(
        description="Fit the ray-traced FSSS displacement-to-gradient calibration."
    )
    parser.add_argument(
        "--metadata",
        default="inputs/calibration_metadata.yaml",
        help="Calibration metadata YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Calibration output directory. Defaults to outputs/full_fsss/calibration/<calibration_id>.",
    )
    parser.add_argument("--max-dots", type=int, default=350, help="Maximum calibration dots used for ray tracing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # calibration metadata contains geometry, refractive indices, liquid depth,
    # and links to prior calibration outputs.
    METADATA_PATH = project_path(args.metadata)
    if METADATA_PATH is None:
        raise FileNotFoundError("Provide --metadata or create inputs/calibration_metadata.yaml.")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Calibration metadata not found: {METADATA_PATH}\n"
            "Provide --metadata or create inputs/calibration_metadata.yaml."
        )

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    CALIBRATION_ID = metadata["calibration_id"]
    if args.output_dir:
        # explicit output directory is useful when testing a new calibration.
        CALIBRATION_OUTPUT_DIR = project_path(args.output_dir)
        if CALIBRATION_OUTPUT_DIR is None:
            raise ValueError("Could not resolve --output-dir.")
    else:
        CALIBRATION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_fsss" / "calibration" / CALIBRATION_ID
    CALIBRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # required previous outputs: camera intrinsics, dry-grid pose, dry-grid dot
    # reference, flat-liquid dot reference, and flat optical geometry.
    camera_metadata = metadata.get("camera", {})
    intrinsics_declared = camera_metadata.get("intrinsics_path")
    INTRINSICS_PATH = project_path(intrinsics_declared) if intrinsics_declared else CALIBRATION_OUTPUT_DIR / "camera_intrinsics.json"
    POSE_PATH = CALIBRATION_OUTPUT_DIR / "camera_pose_grid.json"
    DOT_REF_PATH = CALIBRATION_OUTPUT_DIR / "dot_grid_reference.csv"
    FLAT_DOTS_PATH = CALIBRATION_OUTPUT_DIR / "flat_liquid_dots.csv"
    OPTICAL_GEOMETRY_PATH = CALIBRATION_OUTPUT_DIR / "optical_geometry_refined.json"

    for p in [INTRINSICS_PATH, POSE_PATH, DOT_REF_PATH, FLAT_DOTS_PATH, OPTICAL_GEOMETRY_PATH]:
        # raytrace calibration cannot run until both previous calibration stages
        # have completed.
        if not p.exists():
            raise FileNotFoundError(f"Missing required previous output: {p}")

    with INTRINSICS_PATH.open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)

    with POSE_PATH.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    with OPTICAL_GEOMETRY_PATH.open("r", encoding="utf-8") as f:
        optical_geometry = json.load(f)

    dot_ref = pd.read_csv(DOT_REF_PATH)
    flat_dots = pd.read_csv(FLAT_DOTS_PATH)

    print(f"Ray-trace FSSS calibration: {CALIBRATION_ID}")
    print(f"Input dots: dry={len(dot_ref)}, flat={len(flat_dots)}")


    MAX_DOTS_FOR_RAYTRACE = int(args.max_dots)
    # copy constants to local names used throughout the nested functions.
    RANDOM_SEED = DEFAULT_RANDOM_SEED
    MAX_NFEV = DEFAULT_MAX_NFEV
    ROOT_TOL_M = DEFAULT_ROOT_TOL_M
    MAX_ACCEPTED_RAY_RESIDUAL_M = DEFAULT_MAX_ACCEPTED_RAY_RESIDUAL_M
    PHASES_PER_SURFACE_MODEL = DEFAULT_PHASES_PER_SURFACE_MODEL
    CENTRAL_FRACTION_OF_DOTS = DEFAULT_CENTRAL_FRACTION_OF_DOTS

    # camera matrix, pose, and optical stack define the world-to-pixel geometry
    # used by every simulated ray.
    # camera matrices and pose.

    # prefer the new camera matrix used for undistorted pixels.
    if "new_camera_matrix_alpha0" in intrinsics:
        # prefer undistorted-pixel camera matrix when available.
        K = np.array(intrinsics["new_camera_matrix_alpha0"], dtype=float)
    elif "camera_matrix" in intrinsics:
        K = np.array(intrinsics["camera_matrix"], dtype=float)
    else:
        raise KeyError("camera_intrinsics.json must contain new_camera_matrix_alpha0 or camera_matrix.")

    R_world_to_camera = np.array(optical_geometry["camera"]["R_world_to_camera"], dtype=float)
    # pose maps world/grid coordinates into camera coordinates.
    t_world_to_camera = np.array(optical_geometry["camera"]["tvec_world_to_camera_m"], dtype=float).reshape(3)
    camera_center_world = np.array(optical_geometry["camera"]["camera_center_world_m"], dtype=float).reshape(3)

    stack = optical_geometry["optical_stack"]

    n_air = float(stack.get("n_air", metadata.get("raytrace", {}).get("air_refractive_index", 1.0)))
    n_liquid = float(stack["n_liquid"])
    z_dot = float(stack["z_dot_plane_m"])
    z_free_nominal = float(stack["z_free_surface_m"])
    z_solid_top = float(stack.get("z_solid_top_m", stack.get("z_inner_bottom_m", z_dot)))
    layers_dot_to_liquid = stack.get("layers_dot_to_liquid", [])

    # compatibility aliases retained for older saved stacks/reports.
    n_bottom = float(stack.get("n_bottom", metadata.get("container", {}).get("bottom_refractive_index", 1.49)))
    include_bottom = bool(stack.get("include_bottom", bool(layers_dot_to_liquid)))
    z_outer_bottom = float(stack.get("z_outer_bottom_m", z_dot))
    z_inner_bottom = float(stack.get("z_inner_bottom_m", z_solid_top))

    # empirical flat-reference homography: simulated and measured pixels are
    # converted back to apparent object-plane coordinates using the same flat
    # liquid mapping that later tracking uses.
    # load the flat-liquid homography from object-plane millimetres to pixels.
    flat_map = optical_geometry["flat_liquid_empirical_mapping"]
    # the flat homography connects object-plane millimetres and processed pixels.
    H_objmm_to_pixel = np.array(flat_map["homography_object_mm_to_pixel"], dtype=float)
    H_pixel_to_objmm = np.linalg.inv(H_objmm_to_pixel)

    def apply_homography(H, xy):
        # generic 2D projective transform helper. It preserves input shape.
        xy = np.asarray(xy, dtype=float)
        original_shape = xy.shape
        xy2 = xy.reshape(-1, 2)
        hom = np.column_stack([xy2, np.ones(len(xy2))])
        out = (H @ hom.T).T
        # homogeneous divide converts [x*w, y*w, w] back to [x, y].
        out = out[:, :2] / out[:, 2:3]
        return out.reshape(original_shape)

    def pixel_to_object_xy_m(uv_px):
        """Map undistorted pixel coordinates to empirical flat-reference object-plane x,y in metres."""
        obj_mm = apply_homography(H_pixel_to_objmm, np.asarray(uv_px, dtype=float))
        return obj_mm * 1e-3

    def object_xy_m_to_pixel(xy_m):
        """Map object-plane x,y in metres to flat-reference undistorted pixel coordinates."""
        xy_mm = np.asarray(xy_m, dtype=float) * 1e3
        return apply_homography(H_objmm_to_pixel, xy_mm)

    # ## Optical ray-tracing functions
    # 
    # the solver below works in the world/grid coordinate system:
    # 
    # - printed dot plane: \(z=0\);
    # - flat liquid-air surface: \(z=z_{\mathrm{free}}\);
    # - camera pose from the dry-grid calibration;
    # - optical stack from the flat-reference calibration.
    # 
    # for a candidate surface point \(Q=(x_Q,y_Q,z_s(x_Q,y_Q))\), the ray travels
    # 
    # \[
    # \text{camera} \rightarrow Q \rightarrow \text{liquid} \rightarrow \text{bath bottom, if included} \rightarrow \text{dot plane}.
    # \]
    # 
    # we solve for the \(Q\) whose traced ray lands on a chosen dot.



    # ray-tracing helpers implement vector normalization, Snell refraction,
    # z-plane intersections, camera projection, and propagation through the
    # configured optical stack to the dot plane.
    def unit(v):
        # normalize a vector while tolerating zero/non-finite norms.
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n == 0 or not np.isfinite(n):
            return np.asarray(v, dtype=float)
        return np.asarray(v, dtype=float) / n

    def refract_direction(I, N_medium1_to_medium2, n1, n2):
        """
        Vector Snell refraction.

        I: unit ray direction in medium 1, pointing toward interface.
        N_medium1_to_medium2: unit normal pointing from medium 1 into medium 2.
        n1, n2: refractive indices.
        """
        I = unit(I)
        N = unit(N_medium1_to_medium2)
        cos_to_normal = float(np.dot(I, N))
        if cos_to_normal <= 0:
            # reverse the normal if supplied in opposite orientation.
            N = -N
            cos_to_normal = float(np.dot(I, N))

        # decompose incident direction into normal and tangential components and
        # scale the tangential part by n1/n2 according to Snell's law.
        I_parallel = I - cos_to_normal * N
        T_parallel = (n1 / n2) * I_parallel
        mag2 = float(np.dot(T_parallel, T_parallel))
        if mag2 > 1.0:
            return None  # total internal reflection
        # normal component magnitude is chosen so output direction has unit norm.
        T_normal = math.sqrt(max(0.0, 1.0 - mag2)) * N
        return unit(T_parallel + T_normal)

    def intersect_z(P, d, z):
        # intersect ray P + s*d with horizontal plane at coordinate z.
        P = np.asarray(P, dtype=float)
        d = np.asarray(d, dtype=float)
        if abs(d[2]) < 1e-15:
            return None
        s = (z - P[2]) / d[2]
        if s < -1e-12:
            # intersection is behind the ray origin.
            return None
        if abs(s) <= 1e-12:
            return P.copy()
        return P + s * d

    def project_world_to_pixel(P_world, K, R, t):
        # standard pinhole projection: world -> camera -> normalized image ->
        # pixels via K.
        P_world = np.asarray(P_world, dtype=float).reshape(3)
        X_cam = R @ P_world + t.reshape(3)
        if X_cam[2] <= 0:
            return None
        uvw = K @ (X_cam / X_cam[2])
        return uvw[:2]

    def _layer_n(layer):
        return float(layer.get("refractive_index", layer.get("n")))

    def trace_down_through_stack(P, d, current_n, stack):
        """Trace a downward ray from the liquid through all solid layers to z=0."""
        # starting at the free surface, propagate downward through flat optical
        # layers until the dot plane z=0 is reached.
        P = np.asarray(P, dtype=float)
        d = unit(d)
        N_down = np.array([0.0, 0.0, -1.0])
        z_dot_local = float(stack["z_dot_plane_m"])

        for layer in reversed(stack.get("layers_dot_to_liquid", [])):
            # layers were stored dot->liquid, so reverse order gives
            # liquid->dot tracing.
            z_top = float(layer["z_top_m"])
            z_bottom = float(layer["z_bottom_m"])
            n_layer = _layer_n(layer)

            P = intersect_z(P, d, z_top)
            if P is None:
                return None, None
            if abs(current_n - n_layer) > 1e-12:
                # refract when crossing into a layer with different refractive
                # index.
                d = refract_direction(d, N_down, current_n, n_layer)
                if d is None:
                    return None, None
            current_n = n_layer

            P = intersect_z(P, d, z_bottom)
            if P is None:
                return None, None

        if P[2] > z_dot_local + 1e-12:
            d = refract_direction(d, N_down, current_n, float(stack["n_air"]))
            if d is None:
                return None, None
        P = intersect_z(P, d, z_dot_local)
        if P is None:
            return None, None
        return P, d

    def trace_from_surface_point_to_dot_plane(Q, normal_air_to_liquid, K, R, t, stack):
        """
        Given a surface point Q and normal from air to liquid, trace the ray from camera to Q,
        refract into liquid, through all configured flat layers, and intersect the dot plane.
        Returns the hit point on z=0 and the pixel corresponding to the air-side ray.
        """
        Q = np.asarray(Q, dtype=float).reshape(3)

        # air-side direction from camera center to surface point.
        d_air = unit(Q - camera_center_world)

        # refract air -> liquid at curved free surface.
        d = refract_direction(d_air, normal_air_to_liquid, n_air, n_liquid)
        if d is None:
            return None, None

        P, d = trace_down_through_stack(Q.copy(), d, n_liquid, stack)
        if P is None:
            return None, None

        uv = project_world_to_pixel(Q, K, R, t)
        # return both the dot-plane hit and the camera pixel corresponding to Q.
        return P, uv


    # ## surface models
    # 
    # the calibration uses known artificial surfaces. For each artificial surface, the true slope is known analytically.
    # 
    # a one-dimensional sinusoidal model is
    # 
    # \[
    # \zeta(x,y) = A\cos(k\,s+\phi),
    # \]
    # 
    # where \(s\) is \(x\), \(y\), or a diagonal coordinate. The true slope is
    # 
    # \[
    # \nabla \zeta = -Ak\sin(k\,s+\phi)\,\nabla s.
    # \]



    # synthetic surface models provide known analytic gradients. The raytrace
    # simulation measures apparent dot displacement caused by those known slopes,
    # which is what makes the displacement-to-gradient fit possible.
    def normalize_direction_name(direction):
        # normalize several user-friendly names into the four supported 1D
        # sinusoid directions.
        direction = str(direction).lower().strip()
        if direction in ["x", "horizontal"]:
            return "x"
        if direction in ["y", "vertical"]:
            return "y"
        if direction in ["diagonal", "diag", "xy"]:
            return "diagonal"
        if direction in ["anti_diagonal", "antidiagonal", "xminusy"]:
            return "anti_diagonal"
        raise ValueError(f"Unknown surface-model direction: {direction}")

    def surface_direction_vector(direction):
        # unit vector in the spatial direction of the sinusoidal phase.
        direction = normalize_direction_name(direction)
        if direction == "x":
            return np.array([1.0, 0.0])
        if direction == "y":
            return np.array([0.0, 1.0])
        if direction == "diagonal":
            return unit(np.array([1.0, 1.0]))
        if direction == "anti_diagonal":
            return unit(np.array([1.0, -1.0]))
        raise ValueError(direction)

    def surface_z_and_grad(x, y, model):
        """
        Return z_surface and gradient dz/dx, dz/dy at x,y in metres.
        """
        if model["type"] == "flat":
            # flat reference surface has zero gradient.
            return z_free_nominal, np.array([0.0, 0.0], dtype=float)

        if model["type"] != "sinusoidal_1d":
            raise ValueError(f"Unsupported surface model: {model}")

        A = float(model["amplitude_m"])
        lam = float(model["wavelength_m"])
        phase = float(model.get("phase_rad", 0.0))
        dvec = surface_direction_vector(model.get("direction", "x"))
        k = 2.0 * np.pi / lam

        s = dvec[0] * x + dvec[1] * y
        arg = k * s + phase
        # eta is height perturbation about the nominal free surface.
        eta = A * np.cos(arg)
        # analytic gradient of A cos(k s + phase).
        grad_eta = -A * k * np.sin(arg) * dvec
        return z_free_nominal + eta, grad_eta

    def surface_normal_air_to_liquid(x, y, model):
        # surface normal from liquid to air for z = zeta(x,y) is proportional to
        # [-dz/dx, -dz/dy, 1]. We return the opposite direction: air -> liquid.
        z, grad = surface_z_and_grad(x, y, model)
        dzdx, dzdy = grad
        n_liquid_to_air = unit(np.array([-dzdx, -dzdy, 1.0]))
        return -n_liquid_to_air




    # expand metadata surface models with several phases. Multiple directions
    # and phases improve conditioning of the 2D least-squares fit for B.
    # build surface model list from metadata and expand with phases.
    metadata_models = metadata.get("raytrace", {}).get("calibration_surface_models", [])

    models_base = []
    for m in metadata_models:
        if m.get("type") != "sinusoidal_1d":
            continue
        # convert metadata amplitude/wavelength from mm to SI meters.
        models_base.append({
            "type": "sinusoidal_1d",
            "direction": normalize_direction_name(m.get("direction", "x")),
            "amplitude_m": float(m.get("amplitude_mm", 0.2)) * 1e-3,
            "wavelength_m": float(m.get("wavelength_mm", 8.0)) * 1e-3,
            "source": "metadata",
        })

    if not USE_ONLY_METADATA_SURFACE_MODELS:
        # add additional non-identical models for a better conditioned 2D fit.
        models_base.extend([
            {"type": "sinusoidal_1d", "direction": "anti_diagonal", "amplitude_m": 0.20e-3, "wavelength_m": 9.3e-3, "source": "auto"},
            {"type": "sinusoidal_1d", "direction": "x", "amplitude_m": 0.15e-3, "wavelength_m": 6.7e-3, "source": "auto"},
            {"type": "sinusoidal_1d", "direction": "y", "amplitude_m": 0.15e-3, "wavelength_m": 7.4e-3, "source": "auto"},
        ])

    models = []
    for idx, m in enumerate(models_base):
        for phase in PHASES_PER_SURFACE_MODEL:
            # each physical sinusoid is repeated at several phases so the same
            # dots sample a range of positive/negative slopes.
            mm = dict(m)
            mm["phase_rad"] = float(phase)
            mm["model_id"] = f"model_{len(models):03d}_{mm['direction']}_A{mm['amplitude_m']*1e3:.3f}mm_L{mm['wavelength_m']*1e3:.2f}mm_phi{phase:.2f}"
            models.append(mm)

    if not models:
        raise RuntimeError("No valid calibration surface models found.")

    # ## Select calibration dots
    # 
    # use the same flat-liquid reference dots that fsss tracking will use. The sampling is limited by
    # `MAX_DOTS_FOR_RAYTRACE` to keep runtime manageable.
    # 
    # the calibration should usually use the central region, because edge dots are more affected by walls, menisci, container distortion, and imperfect bottom flatness.



    # calibration-dot sampling uses the flat-liquid dots that later fsss tracking
    # will use, restricted to the central region and a max count for runtime.
    # use flat-liquid dots because these are known to be trackable through the liquid.
    calib_dots = flat_dots.copy()

    # ensure required columns.
    required_cols = {"dot_id", "x_grid_m", "y_grid_m", "x_px", "y_px"}
    missing = required_cols - set(calib_dots.columns)
    if missing:
        raise ValueError(f"flat_liquid_dots.csv is missing required columns: {missing}")

    # central filtering.
    # edge dots are more likely to be affected by walls, meniscus, and poor
    # optical geometry, so fit from a central fraction first.
    x = calib_dots["x_grid_m"].to_numpy(float)
    y = calib_dots["y_grid_m"].to_numpy(float)
    x0, y0 = np.median(x), np.median(y)
    rx = (np.max(x) - np.min(x)) * CENTRAL_FRACTION_OF_DOTS / 2
    ry = (np.max(y) - np.min(y)) * CENTRAL_FRACTION_OF_DOTS / 2

    central = calib_dots[
        (np.abs(calib_dots["x_grid_m"] - x0) <= rx) &
        (np.abs(calib_dots["y_grid_m"] - y0) <= ry)
    ].copy()

    if len(central) < 50:
        central = calib_dots.copy()

    if len(central) > MAX_DOTS_FOR_RAYTRACE:
        # runtime control: sample a reproducible subset.
        central = central.sample(MAX_DOTS_FOR_RAYTRACE, random_state=RANDOM_SEED).copy()

    central = central.sort_values("dot_id").reset_index(drop=True)

    print(f"Calibration dots selected: {len(central)}")
    print(f"Synthetic surface models: {len(models)}")


    # ## solve the ray geometry
    # 
    # for each dot, the solver finds the free-surface point \(Q\) such that the ray from the camera, after refraction at \(Q\), lands on that printed dot.
    # 
    # for the flat surface, this provides \(Q_{\mathrm{flat}}\) and the simulated flat pixel.
    # 
    # for each synthetic curved surface, the solver provides \(Q_{\mathrm{curved}}\), the simulated curved pixel, the apparent object-plane displacement, and the known true slope at \(Q_{\mathrm{curved}}\).



    # surface-point solve: find q=(x,y) on the candidate free surface whose
    # refracted ray lands on the selected printed dot. This is the nonlinear
    # least-squares core of the ray-trace calibration.
    def hit_residual_for_qxy(qxy, dot_xy, model):
        """
        Residual [hit_x - dot_x, hit_y - dot_y] in metres for a candidate surface point qxy.
        """
        qx, qy = float(qxy[0]), float(qxy[1])
        # candidate Q lies on the synthetic free surface at x=qx,y=qy.
        z, grad = surface_z_and_grad(qx, qy, model)

        # safety check: surface must remain above the top solid/liquid interface.
        if z <= z_solid_top + 1e-8:
            return np.array([1e2, 1e2], dtype=float)

        N_air_to_liquid = surface_normal_air_to_liquid(qx, qy, model)
        Q = np.array([qx, qy, z], dtype=float)
        # trace camera -> Q -> refracted stack -> dot plane.
        hit, uv = trace_from_surface_point_to_dot_plane(Q, N_air_to_liquid, K, R_world_to_camera, t_world_to_camera, stack)
        if hit is None or uv is None or not np.all(np.isfinite(hit)):
            return np.array([1e2, 1e2], dtype=float)
        return hit[:2] - np.asarray(dot_xy, dtype=float)

    def solve_q_for_dot(dot_xy, model, q0_xy):
        """
        Solve for the surface point qxy that images a given dot.
        """
        # least_squares varies qx,qy until the traced hit point equals the chosen
        # dot coordinates on z=0.
        res = least_squares(
            lambda q: hit_residual_for_qxy(q, dot_xy, model),
            x0=np.asarray(q0_xy, dtype=float),
            xtol=ROOT_TOL_M,
            ftol=ROOT_TOL_M,
            gtol=ROOT_TOL_M,
            max_nfev=MAX_NFEV,
        )
        qxy = res.x
        final_res = hit_residual_for_qxy(qxy, dot_xy, model)
        # accept only converged solves whose dot-plane residual is below the
        # configured physical tolerance.
        res_norm = float(np.linalg.norm(final_res))
        success = bool(res.success and np.isfinite(res_norm) and res_norm <= MAX_ACCEPTED_RAY_RESIDUAL_M)
        return {
            "success": success,
            "qxy": qxy,
            "residual_m": res_norm,
            "nfev": int(res.nfev),
            "message": str(res.message),
        }

    def simulate_dot_for_model(dot_xy, model, q0_xy):
        # solve surface point Q for one dot and one surface model, then return
        # the camera pixel and analytic gradient at Q.
        sol = solve_q_for_dot(dot_xy, model, q0_xy)
        if not sol["success"]:
            return None

        qx, qy = sol["qxy"]
        z, grad = surface_z_and_grad(qx, qy, model)
        N_air_to_liquid = surface_normal_air_to_liquid(qx, qy, model)
        Q = np.array([qx, qy, z], dtype=float)
        hit, uv = trace_from_surface_point_to_dot_plane(Q, N_air_to_liquid, K, R_world_to_camera, t_world_to_camera, stack)

        if hit is None or uv is None:
            return None

        return {
            "Q": Q,
            "uv": np.asarray(uv, dtype=float),
            "hit": np.asarray(hit, dtype=float),
            "grad": np.asarray(grad, dtype=float),
            "residual_m": sol["residual_m"],
            "nfev": sol["nfev"],
        }




    # flat-surface solve establishes the reference ray solution for each dot and
    # lets us compare simulated flat pixels with the observed flat-liquid pixels.
    # first solve the flat surface for each selected dot. Cache results.
    flat_model = {"type": "flat", "model_id": "flat"}

    flat_solutions = {}
    flat_rows = []

    for idx, r in central.iterrows():
        dot_id = int(r["dot_id"])
        dot_xy = np.array([float(r["x_grid_m"]), float(r["y_grid_m"])])
        q0 = dot_xy.copy()  # good initial guess for near-normal imaging

        # flat model gives the reference simulated ray for this dot.
        sim = simulate_dot_for_model(dot_xy, flat_model, q0)
        if sim is None:
            flat_rows.append({
                "dot_id": dot_id,
                "success": False,
                "residual_m": np.nan,
            })
            continue

        flat_solutions[dot_id] = sim
        flat_rows.append({
            "dot_id": dot_id,
            "success": True,
            "x_grid_m": dot_xy[0],
            "y_grid_m": dot_xy[1],
            "q_flat_x_m": sim["Q"][0],
            "q_flat_y_m": sim["Q"][1],
            "q_flat_z_m": sim["Q"][2],
            "u_flat_sim_px": sim["uv"][0],
            "v_flat_sim_px": sim["uv"][1],
            "residual_m": sim["residual_m"],
            "nfev": sim["nfev"],
        })

    flat_solution_df = pd.DataFrame(flat_rows)
    n_flat_ok = int(flat_solution_df["success"].sum())
    if n_flat_ok < max(25, len(flat_solution_df) // 3):
        raise RuntimeError("Too few successful flat ray solutions. Check optical geometry and stack parameters.")

    # compare simulated flat pixels to observed flat-liquid reference pixels.
    flat_ok = flat_solution_df[flat_solution_df["success"]].copy()
    # merge simulated flat pixels with measured flat-liquid pixels to quantify
    # how well the physical ray model reproduces the empirical flat reference.
    flat_merge = flat_ok.merge(
        central[["dot_id", "x_px", "y_px", "x_grid_m", "y_grid_m"]],
        on="dot_id",
        how="left",
        suffixes=("_simrow", "_obs")
    )

    flat_merge["du_sim_minus_obs_px"] = flat_merge["u_flat_sim_px"] - flat_merge["x_px"]
    flat_merge["dv_sim_minus_obs_px"] = flat_merge["v_flat_sim_px"] - flat_merge["y_px"]
    flat_merge["flat_pixel_residual_px"] = np.sqrt(flat_merge["du_sim_minus_obs_px"]**2 + flat_merge["dv_sim_minus_obs_px"]**2)

    # ## Curved-surface simulations and matrix fit
    # 
    # for every successful dot and every synthetic surface, compute:
    # 
    # \[
    # \Delta \mathbf r_{\mathrm{obj}}
    # =
    # \mathbf r_{\mathrm{obj,curved}}
    # -
    # \mathbf r_{\mathrm{obj,flat}},
    # \]
    # 
    # \[
    # \mathbf g
    # =
    # \begin{pmatrix}
    # \partial_x \zeta\\
    # \partial_y \zeta
    # \end{pmatrix}_{Q_{\mathrm{curved}}}.
    # \]
    # 
    # then it solves the least-squares problem
    # 
    # \[
    # \mathbf g = \mathbf B \Delta \mathbf r_{\mathrm{obj}}.
    # \]



    # curved-surface simulations produce paired samples: apparent object-plane
    # displacement from the empirical homography and the known analytic surface
    # gradient at the solved curved surface point.
    rows = []

    for model_index, model in enumerate(models):
        for _, r in central.iterrows():
            dot_id = int(r["dot_id"])
            if dot_id not in flat_solutions:
                continue

            dot_xy = np.array([float(r["x_grid_m"]), float(r["y_grid_m"])])
            flat_sim = flat_solutions[dot_id]
            # start curved solve from the already solved flat surface point.
            q0 = flat_sim["Q"][:2]

            curved = simulate_dot_for_model(dot_xy, model, q0)
            if curved is None:
                rows.append({
                    "model_id": model["model_id"],
                    "dot_id": dot_id,
                    "success": False,
                    "reason": "curved_solve_failed",
                })
                continue

            # object-plane coordinates from empirical flat-liquid homography.
            uv_flat = flat_sim["uv"]
            uv_curved = curved["uv"]
            # convert both simulated pixels back through the same empirical flat
            # mapping used for measured tracks.
            obj_flat = pixel_to_object_xy_m(uv_flat)
            obj_curved = pixel_to_object_xy_m(uv_curved)
            delta_obj = obj_curved - obj_flat

            Q_flat = flat_sim["Q"]
            Q_curved = curved["Q"]
            delta_surface_point = Q_curved[:2] - Q_flat[:2]

            # this pair is the calibration sample:
            # input = apparent object displacement, output = true surface slope.
            grad = curved["grad"]
            slope_mag = float(np.linalg.norm(grad))
            delta_mag = float(np.linalg.norm(delta_obj))

            rows.append({
                "model_id": model["model_id"],
                "model_direction": model.get("direction", ""),
                "model_amplitude_m": float(model["amplitude_m"]),
                "model_wavelength_m": float(model["wavelength_m"]),
                "model_phase_rad": float(model.get("phase_rad", 0.0)),
                "model_source": model.get("source", ""),
                "dot_id": dot_id,
                "success": True,
                "x_grid_m": dot_xy[0],
                "y_grid_m": dot_xy[1],
                "q_flat_x_m": Q_flat[0],
                "q_flat_y_m": Q_flat[1],
                "q_curved_x_m": Q_curved[0],
                "q_curved_y_m": Q_curved[1],
                "q_curved_z_m": Q_curved[2],
                "u_flat_sim_px": uv_flat[0],
                "v_flat_sim_px": uv_flat[1],
                "u_curved_sim_px": uv_curved[0],
                "v_curved_sim_px": uv_curved[1],
                "du_px": uv_curved[0] - uv_flat[0],
                "dv_px": uv_curved[1] - uv_flat[1],
                "dx_obj_m": delta_obj[0],
                "dy_obj_m": delta_obj[1],
                "delta_obj_magnitude_m": delta_mag,
                "surface_point_dx_m": delta_surface_point[0],
                "surface_point_dy_m": delta_surface_point[1],
                "grad_x": grad[0],
                "grad_y": grad[1],
                "slope_magnitude": slope_mag,
                "curved_solve_residual_m": curved["residual_m"],
                "curved_solve_nfev": curved["nfev"],
            })

    calib_samples = pd.DataFrame(rows)
    ok_samples = calib_samples[calib_samples["success"]].copy()
    if len(ok_samples) < 100:
        raise RuntimeError("Too few successful curved-surface simulations. Check geometry, reduce model amplitudes, or increase solver tolerance.")

    # fit B: gradient = B @ displacement_object.
    X = ok_samples[["dx_obj_m", "dy_obj_m"]].to_numpy(float)
    Y = ok_samples[["grad_x", "grad_y"]].to_numpy(float)

    # remove rows with almost zero displacement because they provide little information.
    # tiny X values make the inverse calibration ill-conditioned.
    disp_mag = np.linalg.norm(X, axis=1)
    slope_mag = np.linalg.norm(Y, axis=1)
    min_disp = max(1e-9, np.percentile(disp_mag, 5) * 0.25)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1) & (disp_mag > min_disp)

    X_fit = X[mask]
    Y_fit = Y[mask]

    # displacement-to-gradient matrix B: solve Y = X @ B.T without an intercept,
    # because zero apparent displacement should correspond to zero slope.
    # least-squares without intercept: Y = X @ B_T, so B = B_T.T.
    # np.linalg.lstsq returns B_T minimizing ||X_fit @ B_T - Y_fit||.
    B_T, residuals, rank, svals = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
    B = B_T.T

    Y_pred = X_fit @ B_T
    res = Y_fit - Y_pred
    rmse_grad = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
    rmse_grad_x = float(np.sqrt(np.mean(res[:, 0]**2)))
    rmse_grad_y = float(np.sqrt(np.mean(res[:, 1]**2)))

    # surface-point correction L estimates how the actual probed surface point
    # shifts relative to the flat-surface point when dots are displaced.
    # fit surface-point correction matrix L: delta_Qxy = L @ delta_obj.
    Yq = ok_samples[["surface_point_dx_m", "surface_point_dy_m"]].to_numpy(float)
    Xq = ok_samples[["dx_obj_m", "dy_obj_m"]].to_numpy(float)
    mask_q = np.isfinite(Xq).all(axis=1) & np.isfinite(Yq).all(axis=1) & (np.linalg.norm(Xq, axis=1) > min_disp)

    # L is fitted with the same no-intercept least-squares form:
    # delta_Qxy = L @ delta_obj.
    L_T, residuals_L, rank_L, svals_L = np.linalg.lstsq(Xq[mask_q], Yq[mask_q], rcond=None)
    L = L_T.T

    Yq_pred = Xq[mask_q] @ L_T
    res_q = Yq[mask_q] - Yq_pred
    rmse_surface_point_m = float(np.sqrt(np.mean(np.sum(res_q**2, axis=1))))

    # what this writes:
    # raytrace_fsss_calibration.json is the main input to metric tracking and
    # surface reconstruction
    # it is what converts object-plane displacement into surface gradient



    # valid ranges from simulation.
    disp_mag_fit = np.linalg.norm(X_fit, axis=1)
    slope_mag_fit = np.linalg.norm(Y_fit, axis=1)
    valid_slope_range = {
        # these ranges are saved so reconstruction can warn when measured runs
        # extrapolate beyond the synthetic calibration regime.
        "max_abs_grad_x": float(np.max(np.abs(Y_fit[:, 0]))),
        "max_abs_grad_y": float(np.max(np.abs(Y_fit[:, 1]))),
        "p95_slope_magnitude": float(np.percentile(slope_mag_fit, 95)),
        "max_slope_magnitude": float(np.max(slope_mag_fit)),
    }

    valid_displacement_range = {
        "p05_delta_obj_magnitude_m": float(np.percentile(disp_mag_fit, 5)),
        "p50_delta_obj_magnitude_m": float(np.percentile(disp_mag_fit, 50)),
        "p95_delta_obj_magnitude_m": float(np.percentile(disp_mag_fit, 95)),
        "max_delta_obj_magnitude_m": float(np.max(disp_mag_fit)),
    }

    calibration = {
        # this json is the main fsss calibration artifact consumed downstream.
        "calibration_id": CALIBRATION_ID,
        "description": (
            "Ray-traced FSSS calibration. Converts apparent object-plane dot displacement "
            "into liquid-surface gradient."
        ),
        "coordinate_system": {
            "description": (
                "World/grid coordinates are attached to the printed dot plane. "
                "z=0 is the dot plane; x,y are in metres."
            ),
            "units": "SI metres unless field name says otherwise",
        },
        "units": {
            "object_displacement": "m",
            "surface_gradient": "dimensionless",
            "B_displacement_to_gradient": "1/m",
            "B_displacement_to_gradient_per_mm": "1/mm",
            "height_reconstruction": "m when x,y are supplied in m",
        },
        "inputs": {
            "metadata": relpath(METADATA_PATH),
            "camera_intrinsics": relpath(INTRINSICS_PATH),
            "camera_pose_grid": relpath(POSE_PATH),
            "dot_grid_reference": relpath(DOT_REF_PATH),
            "flat_liquid_dots": relpath(FLAT_DOTS_PATH),
            "optical_geometry": relpath(OPTICAL_GEOMETRY_PATH),
        },
        "camera": {
            "camera_matrix_used_for_undistorted_pixels": K.tolist(),
            "R_world_to_camera": R_world_to_camera.tolist(),
            "tvec_world_to_camera_m": t_world_to_camera.reshape(3).tolist(),
            "camera_center_world_m": camera_center_world.tolist(),
        },
        "optical_stack": stack,
        "flat_reference_mapping": {
            "homography_object_mm_to_pixel": H_objmm_to_pixel.tolist(),
            "homography_pixel_to_object_mm": H_pixel_to_objmm.tolist(),
            "note": (
                "Object-plane displacement is computed by applying this inverse homography "
                "to simulated or measured undistorted pixel positions."
            ),
        },
        "surface_models_used": models,
        "raytrace_settings": {
            "max_dots_for_raytrace": MAX_DOTS_FOR_RAYTRACE,
            "central_fraction_of_dots": CENTRAL_FRACTION_OF_DOTS,
            "phases_per_surface_model": PHASES_PER_SURFACE_MODEL,
            "max_nfev": MAX_NFEV,
            "root_tol_m": ROOT_TOL_M,
            "max_accepted_ray_residual_m": MAX_ACCEPTED_RAY_RESIDUAL_M,
        },
        "fit": {
            "n_successful_simulated_samples": int(len(ok_samples)),
            "n_samples_used_in_B_fit": int(len(X_fit)),
            "B_displacement_to_gradient_1_per_m": B.tolist(),
            "B_displacement_to_gradient_1_per_mm": (B / 1000.0).tolist(),
            "rank": int(rank),
            "singular_values": [float(x) for x in svals],
            "gradient_rmse": rmse_grad,
            "gradient_rmse_x": rmse_grad_x,
            "gradient_rmse_y": rmse_grad_y,
            "valid_slope_range": valid_slope_range,
            "valid_displacement_range": valid_displacement_range,
        },
        "surface_point_correction": {
            "description": (
                "Approximate mapping from apparent object-plane dot displacement to shift "
                "of the probed surface point relative to the flat-surface probed point."
            ),
            "L_surface_point_correction_dimensionless": L.tolist(),
            "rmse_surface_point_m": rmse_surface_point_m,
            "rmse_surface_point_mm": rmse_surface_point_m * 1000.0,
        },
        "flat_model_vs_observed": {
            "median_pixel_residual_px": float(np.median(flat_merge["flat_pixel_residual_px"])),
            "max_pixel_residual_px": float(np.max(flat_merge["flat_pixel_residual_px"])),
            "note": (
                "Large values mean the physical ray model does not reproduce the observed flat reference. "
                "The B matrix may still be useful as a local calibration, but geometry should be checked."
            ),
        },
        "notes": [
            "This calibration assumes undistorted pixels using the configured camera intrinsics.",
            "If the camera, liquid depth, refractive index, dot grid spacing, or bath geometry changes, regenerate this file.",
            "For final metric reconstruction, use B_displacement_to_gradient_1_per_m.",
            "The relation is linearized over the slope range sampled by the synthetic calibration surfaces.",
        ],
    }

    calibration_path = CALIBRATION_OUTPUT_DIR / "raytrace_fsss_calibration.json"
    with calibration_path.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Flat ray solutions: {n_flat_ok}/{len(flat_solution_df)}")
    print(f"Curved ray simulations: {len(ok_samples)}/{len(calib_samples)}")
    print(f"Gradient RMSE: {rmse_grad:.6g}")
    print(f"Samples used: {len(X_fit)}")
    print(f"Saved {relpath(calibration_path)}")


    # ## How to interpret the output
    # 
    # later scripts should use `raytrace_fsss_calibration.json` as follows:
    # 
    # 1. Track dots in a disturbed-surface video.
    # 2. compute apparent object-plane displacement:
    #    \[
    #    \Delta x_{\mathrm{obj}},\Delta y_{\mathrm{obj}}.
    #    \]
    # 3. convert displacement to metric gradients:
    #    \[
    #    \begin{pmatrix}
    #    \partial_x\zeta\\
    #    \partial_y\zeta
    #    \end{pmatrix}
    #    =
    #    \mathbf B
    #    \begin{pmatrix}
    #    \Delta x_{\mathrm{obj}}\\
    #    \Delta y_{\mathrm{obj}}
    #    \end{pmatrix}.
    #    \]
    # 4. Interpolate those gradients onto a regular grid.
    # 5. Integrate the gradient field to reconstruct
    #    \[
    #    \zeta(x,y,t)
    #    \]
    #    in metres.
    # 
    # if `flat_model_vs_observed` residuals are large, check:
    # 
    # - camera intrinsics;
    # - camera pose from the dry grid;
    # - whether the bath bottom is truly flat;
    # - bottom thickness and refractive index;
    # - water/oil depth;
    # - whether the camera moved between dry, empty-bath, and flat-liquid references.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
