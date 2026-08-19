# Implementation

This document explains how the perspective sculpture tool runs, how the main files relate to each other, what the important functions do, what they return, and why they matter. It also separates out the loss terms and optimization methods used by the current implementation.

## Pipeline Overview

The app is a PySide6 desktop tool for arranging small flat patch meshes so they create different silhouettes or appearances from two perspective cameras.

At a high level:

1. `main.py` starts the Qt app and opens the main window.
2. `ui/main_window.py` builds the scene cameras, viewport, image panels, and controls.
3. The user loads one or two target images in `ui/image_panel.py`.
4. The user initializes patches through `core/initialization.py`.
5. The patches are shown in the OpenGL viewport through `ui/viewport.py`.
6. When optimization starts, `ui/worker.py` runs the optimizer in a background thread.
7. `core/optimizer.py` renders the patches through `core/renderer.py`, computes losses through `core/loss.py`, applies optimization updates, applies post-step constraints, and returns updated mesh snapshots.
8. The UI receives progress updates and refreshes both the main viewport and the camera-preview images.

The central data object is a `Patch` from `core/patch.py`. A patch stores its center, rotation, color, and local spline/control-point parameters. It can convert itself into a renderable mesh for both the differentiable renderer and the viewport.

## File Relationships

### `main.py`

Entry point for the application.

- Configures the OpenGL surface format.
- Creates the Qt application.
- Creates and shows `MainWindow`.
- Starts the Qt event loop.

### `ui/main_window.py`

Top-level coordinator.

- Owns the current target images.
- Owns the current patch list.
- Owns the scene cameras.
- Connects UI controls to initialization, optimization, pause, reset, and export behavior.
- Receives optimization updates from `OptimizationWorker`.

### `ui/controls_panel.py`

Right-side control panel.

- Patch count.
- Initialization mode.
- Device selection.
- Loss type for view 2.
- Learning rate, defaulting to `2.75e-03`.
- Fixed-step or convergence mode.
- Step count, defaulting to `400`, and progress bar.
- Convergence threshold.
- Palette input.
- Adaptive patches (SRD): enable checkbox, patch count penalty slider defaulting to `0.05`, and live active/added/deleted status.
- Run, pause, and reset buttons.

### `ui/image_panel.py`

Left-side image panel.

- Lets the user load target images.
- Shows target images without stretching them.
- Shows software-rendered camera previews below the targets.

### `ui/viewport.py`

Central OpenGL viewport.

- Displays grid, axes, camera frustums, and patch meshes.
- Uses the current `Scene` meshes for drawing.

### `scene/camera.py`

Camera model.

- Stores position, target, up vector, field of view, aspect ratio, near clip, and far clip.
- Produces view and projection matrices.
- Produces frustum line geometry for the viewport.

### `scene/scene.py`

Simple scene data structures.

- `Mesh` stores vertices, faces, color, transform, label, and visibility.
- `Scene` stores cameras and meshes.

### `core/patch.py`

Patch geometry and parameters.

- Defines differentiable patch parameters.
- Converts patches to sampled/extruded meshes.
- Provides local-to-world transforms.

### `core/initialization.py`

Patch initialization.

- Random patch centers inside a 3D box (or the swept volume when available).

### `core/renderer.py`

Differentiable rendering through `nvdiffrast`.

- Builds triangle geometry from patches.
- Renders RGBA images from each camera.

### `core/loss.py`

Basic differentiable losses.

- RGB mean squared error.
- Silhouette loss.
- Masked RGB loss.

### `core/optimizer.py`

Optimization logic.

- Converts images to tensors.
- Fits targets to renderer resolution.
- Creates foreground masks.
- Builds optimizer parameter groups.
- Computes geometric penalties.
- Computes patch count penalty for SRD rewrite scoring.
- Applies Adam updates.
- Applies hard post-step constraints.
- Calls stochastic rewrite descent when enabled.

### `optimizer/srd.py`

Stochastic rewrite descent for adaptive patch counts.

- Samples structural rewrites such as add, delete, restore, and split.
- Scores candidate rewrites with local optimization lookahead.
- Applies compatible rewrites that improve the loss.
- Prints a rewrite log whenever it actually adds, restores, splits, or deletes a patch.
- Tracks active, added, and deleted patch counts for the UI.

### `ui/worker.py`

Background optimization thread.

- Keeps the UI responsive while optimization runs.
- Supports pause and stop requests.
- Emits progress and mesh snapshots back to the main window.

## Runtime Flow

### Startup

`main.py` calls `_configure_opengl()`, creates a `QApplication`, creates `MainWindow`, shows it, and then runs `app.exec()`.

`MainWindow` creates two scene cameras with `_make_scene_cameras()`. View 1 looks toward the origin from the positive Z direction. View 2 looks toward the origin from the positive X direction. These cameras are used consistently by the viewport, camera previews, and optimizer.

### Image Loading

When the user loads an image, `ImagePanel` emits `view1_loaded` or `view2_loaded`.

`MainWindow._on_view1_loaded()` and `MainWindow._on_view2_loaded()` open the file with PIL, convert it to RGB, and store it as a NumPy array. These arrays later become optimization targets.

### Patch Initialization

When the user clicks initialize, `MainWindow._on_initialize()` calls:

```python
initialize_patches(...)
```

from `core/initialization.py`.

The selected mode creates a list of `Patch` objects. The main window then snaps patch colors to the current palette, sends the patches to the viewport, and updates the camera previews.

### Optimization

When the user clicks run, `MainWindow._on_run_optimization()` validates that patches and the first target image exist. It then creates an `OptimizationWorker`.

The worker creates a `SceneOptimizer`, then repeatedly calls:

```python
optimizer.step(step_idx, total_steps)
```

Each step:

1. Renders both camera views.
2. Computes image losses and geometry penalties.
3. Backpropagates through differentiable patch parameters.
4. Applies an Adam optimizer step.
5. Applies hard post-step constraints.
6. Returns a metrics dictionary.

The worker emits the step number, metrics, and mesh snapshot. The main window updates the progress bar, viewport, camera previews, and status text.

## Important Functions By File

## `main.py`

### `_configure_opengl()`

What it does:

- Sets the default Qt OpenGL format before the application window is created.

Returns:

- Nothing.

Why it matters:

- The viewport uses modern OpenGL drawing. This function requests an OpenGL 3.3 core profile and multisampling so the viewport can render consistently.

How it works:

- Creates a `QSurfaceFormat`.
- Sets version, profile, depth buffer size, and sample count.
- Installs it with `QSurfaceFormat.setDefaultFormat()`.

### `main()`

What it does:

- Starts the desktop app.

Returns:

- The Qt process exit code.

Why it matters:

- This is the application entry point.

How it works:

- Configures OpenGL.
- Creates `QApplication`.
- Creates and shows `MainWindow`.
- Runs `app.exec()`.

## `ui/main_window.py`

### `_make_scene_cameras()`

What it does:

- Creates the two perspective cameras used by the whole tool.

Returns:

- A list of two `Camera` objects.

Why it matters:

- These cameras define the two views that the sculpture is optimized against.

How it works:

- Builds one camera on the positive Z axis and one on the positive X axis.
- Both look at the origin.
- Both share field of view, aspect ratio, near clip, and far clip settings.

### `MainWindow.__init__()`

What it does:

- Builds the whole application window.

Returns:

- A constructed `MainWindow` instance.

Why it matters:

- This is where UI panels, scene state, signals, and callbacks are connected.

How it works:

- Creates a `Scene`.
- Creates `ImagePanel`, `Viewport`, and `ControlsPanel`.
- Arranges them in a horizontal splitter.
- Connects image, control, optimization, pause, reset, and export signals.

### `_on_view1_loaded(path)` and `_on_view2_loaded(path)`

What they do:

- Load target image files.

Returns:

- Nothing.

Why they matter:

- Optimization needs target images as RGB arrays.

How they work:

- Use PIL to open the selected path.
- Convert the image to RGB.
- Store it in `_target1` or `_target2`.

### `_on_initialize(n_patches, mode)`

What it does:

- Creates the initial patch layout.

Returns:

- Nothing.

Why it matters:

- The optimizer needs a starting patch configuration. Different starting configurations can produce very different results.

How it works:

- Calls `initialize_patches()`.
- Passes the patch count, cameras, device, and swept volume.
- Snaps patch colors to the palette.
- Sends patches to the viewport.
- Updates camera preview renders.

### `_on_run_optimization()`

What it does:

- Starts the optimization thread.

Returns:

- Nothing.

Why it matters:

- This bridges UI settings into the optimizer.

How it works:

- Validates that patches and a view 1 target exist.
- Reads controls such as learning rate, steps, convergence threshold, palette, and loss mode.
- Creates `OptimizationWorker`.
- Connects worker signals.
- Starts the thread.

### `_on_optimization_step(step, metrics, meshes)`

What it does:

- Handles one optimization update from the worker.

Returns:

- Nothing.

Why it matters:

- Keeps the UI synchronized with the background optimizer.
- Prints the total loss plus per-term average losses for debugging on the same cadence as the existing progress log.

How it works:

- Sends mesh snapshots to the viewport.
- Updates camera previews.
- Updates the fixed-step progress bar.
- Logs periodic total loss, view RGB losses, full per-view totals, raw per-term averages, and weighted geometric penalty contributions.
- Includes separate view 1 and view 2 negative-space losses in the debug output.

### `_on_pause_optimization(paused)`

What it does:

- Pauses or resumes the optimization worker.

Returns:

- Nothing.

Why it matters:

- Lets the user inspect the current state without killing the optimization.

How it works:

- Calls `OptimizationWorker.set_paused(paused)`.

### `_on_reset()`

What it does:

- Resets the whole app state.

Returns:

- Nothing.

Why it matters:

- Gives the user a clean restart without relaunching the app.

How it works:

- Stops the worker if needed.
- Calls `_reset_state()`.

### `_reset_state()`

What it does:

- Clears patches, target images, previews, viewport meshes, controls, and export state.

Returns:

- Nothing.

Why it matters:

- Centralizes reset behavior so the UI and internal state do not drift apart.

How it works:

- Clears stored arrays and patch lists.
- Calls reset methods on image panel, viewport, and controls.

### `_update_camera_previews_from_patches()`

What it does:

- Refreshes the small camera-preview images.

Returns:

- Nothing.

Why it matters:

- Lets the user see what each perspective camera currently sees.

How it works:

- Converts each patch to a `Mesh`.
- Passes meshes and cameras to `ImagePanel.set_camera_previews()`.

## `ui/worker.py`

### `OptimizationWorker.__init__(...)`

What it does:

- Stores all settings needed to run optimization in a background thread.

Returns:

- A constructed worker instance.

Why it matters:

- Keeps long-running optimization off the UI thread.

How it works:

- Saves patches, cameras, targets, palette, learning rate, steps, run mode, threshold, loss mode, and device.

### `request_stop()`

What it does:

- Requests that optimization stop.

Returns:

- Nothing.

Why it matters:

- Allows reset or app shutdown to end the worker cleanly.

How it works:

- Sets an internal stop flag.
- Also clears pause state so a paused worker can exit.

### `set_paused(paused)`

What it does:

- Changes pause state.

Returns:

- Nothing.

Why it matters:

- Enables pause/resume without discarding optimizer state.

How it works:

- Stores the pause flag.

### `_wait_if_paused()`

What it does:

- Blocks the worker loop while paused.

Returns:

- Nothing.

Why it matters:

- Implements pause behavior between optimization steps.

How it works:

- Sleeps briefly in a loop while paused.
- Exits if a stop request arrives.

### `run()`

What it does:

- Runs the optimization loop.

Returns:

- Nothing directly. It communicates through Qt signals.

Why it matters:

- This is the threaded execution path for optimization.

How it works:

- Creates a `SceneOptimizer`.
- Passes the SRD enable state and patch count penalty from the controls panel.
- In fixed-step mode, runs for the selected number of steps.
- In convergence mode, runs until the loss reaches the threshold or the worker is stopped.
- Emits `step_completed`, `failed`, and `optimization_finished` signals.

## `core/initialization.py`

### `_make_patch(center, theta, radius, albedo, device, label)`

What it does:

- Creates one patch with a circular-ish local shape.

Returns:

- A `Patch`.

Why it matters:

- Initialization uses this helper to build consistent patch geometry.

How it works:

- Places several control points around a local circle.
- Creates a patch with a center, rotation angle, albedo color, and label.

### `initialize_patches(...)`

What it does:

- Creates randomly positioned patches.

Returns:

- A list of `Patch` objects.

Why it matters:

- This is the single initialization API used by the UI.

How it works:

- Samples patch centers inside a 3D box, or from the swept volume when one is provided.
- Samples theta only from bands at least 15 degrees away from either camera yaw.
- Chooses an adaptive patch radius based on patch count and box volume.

## `core/patch.py`

### `ControlPoint.__init__(...)`

What it does:

- Creates one differentiable local control point.

Returns:

- A `ControlPoint`.

Why it matters:

- Patch shape is defined by these points.

How it works:

- Stores local x, y, z, handle scale, and handle rotation as torch tensors.

### `ControlPoint.pos`

What it does:

- Returns the control point local position.

Returns:

- A tensor shaped like `[3]`.

Why it matters:

- Spline sampling and mesh generation need point positions as tensors.

How it works:

- Stacks `x`, `y`, and `z`.

### `ControlPoint.handle_out()` and `ControlPoint.handle_in()`

What they do:

- Compute outgoing and incoming Bezier-style handles.

Returns:

- Local handle position tensors.

Why they matter:

- They control the smoothness and curvature of patch outlines.

How they work:

- Use handle scale and handle rotation to create an offset direction.
- Add or subtract that offset from the control point position.

### `Patch.__init__(...)`

What it does:

- Creates a patch object.

Returns:

- A `Patch`.

Why it matters:

- Patch is the core optimization primitive.

How it works:

- Stores center, theta, albedo, control points, thickness, label, and device.

### `Patch.rotation_matrix()`

What it does:

- Builds the local-to-world rotation matrix.

Returns:

- A `3x3` tensor.

Why it matters:

- This determines patch orientation in the scene.

How it works:

- Uses `theta` to rotate the patch around its local axis.

### `Patch.local_to_world(local_points)`

What it does:

- Converts local patch points into world coordinates.

Returns:

- A tensor of world-space points.

Why it matters:

- Rendering and viewport drawing both need world-space geometry.

How it works:

- Applies the patch rotation matrix.
- Adds the patch center.

### `Patch.sample_spline_local(samples_per_segment)`

What it does:

- Samples the patch outline in local space.

Returns:

- A tensor of local vertices.

Why it matters:

- Converts editable control points into a polygonal outline.

How it works:

- Iterates over neighboring control points.
- Evaluates cubic Bezier segments using the outgoing and incoming handles.

### `Patch.sample_spline_world(samples_per_segment)`

What it does:

- Samples the patch outline in world space.

Returns:

- A tensor of world vertices.

Why it matters:

- Used by mesh generation, renderer geometry, and penalties.

How it works:

- Calls `sample_spline_local()`.
- Passes the result through `local_to_world()`.

### `Patch.compute_area(samples_per_segment)`

What it does:

- Computes the enclosed local spline area.

Returns:

- A scalar torch tensor.

Why it matters:

- SRD uses patch area to sample larger patches for split candidates and to identify near-zero-area patches for delete candidates.

How it works:

- Samples the local spline outline.
- Applies the shoelace formula to the sampled XY points.
- Returns the absolute area so clockwise and counter-clockwise point order both work.

### `Patch.split_down_middle(creation_step)`

What it does:

- Creates two five-control-point child patches from one parent patch.

Returns:

- A tuple of `(child_a, child_b)` patches.

Why it matters:

- SRD can evaluate split rewrites when a target needs more local detail than one large patch can represent.

How it works:

- Finds the top-most control point.
- Finds the midpoint between the two lowest control points.
- Builds a split line between those two locations with a center control point.
- Constructs left and right five-point child patches with inherited center, theta, color, and creation step.

### `Patch.world_vertices_homogeneous(samples_per_segment)`

What it does:

- Produces homogeneous world vertices.

Returns:

- A tensor shaped like `[N, 4]`.

Why it matters:

- Camera projection math uses homogeneous coordinates.

How it works:

- Samples world vertices.
- Appends a `1` coordinate to each vertex.

### `Patch.extruded_mesh_world(samples_per_segment)`

What it does:

- Creates a thin 3D mesh from the flat patch outline.

Returns:

- A tuple of `(vertices, faces)`.

Why it matters:

- The renderer and viewport need triangle meshes, not abstract splines.

How it works:

- Samples the outline.
- Creates front and back vertices separated by patch thickness.
- Adds front, back, and side faces.

### `Patch.triangle_faces(n_outline)`

What it does:

- Creates triangle indices for an extruded outline.

Returns:

- A tensor of triangle faces.

Why it matters:

- Rasterization requires triangle indices.

How it works:

- Fans the front and back faces around the first outline vertex.
- Adds side triangles between corresponding front and back outline edges.

### `Patch.to_mesh(samples_per_segment)`

What it does:

- Converts a patch into a viewport `Mesh`.

Returns:

- A `scene.scene.Mesh`.

Why it matters:

- Bridges optimized patch data into the OpenGL viewport and software previews.

How it works:

- Calls `extruded_mesh_world()`.
- Converts torch tensors to NumPy arrays.
- Uses the patch albedo as mesh color.

### `Patch.clamp_albedo()`

What it does:

- Clamps patch color values.

Returns:

- Nothing.

Why it matters:

- Keeps colors in valid RGB range.

How it works:

- Clamps `albedo` in-place to `[0, 1]`.

### `Patch.to_dict()` and `Patch.from_dict(data)`

What they do:

- Serialize and deserialize patch data.

Returns:

- `to_dict()` returns a Python dictionary.
- `from_dict()` returns a `Patch`.

Why they matter:

- Useful for saving, loading, and debugging patch states.

How they work:

- Convert tensor values to plain Python lists and floats.
- Rebuild tensors and control points when loading.
- Preserve `creation_step` so SRD cooldown behavior survives tentative rewrite save/restore passes.

## `scene/camera.py`

### `Camera.view_matrix()`

What it does:

- Computes the camera view matrix.

Returns:

- A `4x4` NumPy matrix.

Why it matters:

- Converts world coordinates into camera/view coordinates.

How it works:

- Computes a look-at basis from position, target, and up vectors.
- Fills rotation and translation into a matrix.

### `Camera.projection_matrix()`

What it does:

- Computes the perspective projection matrix.

Returns:

- A `4x4` NumPy matrix.

Why it matters:

- Converts camera-space coordinates into clip-space coordinates.

How it works:

- Uses field of view, aspect ratio, near clip, and far clip.

### `Camera.frustum_line_vertices()`

What it does:

- Produces line vertices for drawing the camera frustum.

Returns:

- A NumPy array of line endpoint vertices.

Why it matters:

- Lets the viewport show where each camera is looking.

How it works:

- Computes near/far plane corners and connects them as line segments.

## `scene/scene.py`

### `Mesh.bounds()`

What it does:

- Computes the mesh bounding box.

Returns:

- A tuple of `(min_xyz, max_xyz)`.

Why it matters:

- Useful for framing, export, and debugging.

How it works:

- Takes per-axis min and max over mesh vertices.

### `Scene.add_mesh(mesh)`, `remove_mesh(mesh)`, `clear_meshes()`, and `set_meshes(meshes)`

What they do:

- Manage scene mesh contents.

Returns:

- Nothing.

Why they matter:

- The viewport renders whatever meshes are currently stored in the scene.

How they work:

- Append, remove, clear, or replace the scene mesh list.

## `core/renderer.py`

### `DiffRenderer.__init__(device, resolution)`

What it does:

- Creates a differentiable renderer wrapper.

Returns:

- A `DiffRenderer`.

Why it matters:

- Optimization needs rendered pixels that support gradients.

How it works:

- Stores device and resolution.
- Creates an `nvdiffrast` rasterization context.

### `_create_context()`

What it does:

- Creates the `nvdiffrast` rendering context.

Returns:

- A rasterization context object.

Why it matters:

- All differentiable rendering depends on this context.

How it works:

- Uses CUDA or OpenGL context creation depending on availability and device.

### `_camera_mvp(camera)`

What it does:

- Builds a model-view-projection matrix for a camera.

Returns:

- A torch `4x4` matrix.

Why it matters:

- Projects world-space patch vertices into clip space.

How it works:

- Multiplies the camera projection matrix by the camera view matrix.
- Converts the result to a tensor on the renderer device.

### `_build_geometry(patches, samples_per_segment)`

What it does:

- Converts all patches into one batched triangle mesh.

Returns:

- Vertex positions, triangle indices, and per-vertex colors.

Why it matters:

- `nvdiffrast` renders triangle buffers, so patch objects must be packed into tensors.

How it works:

- Calls each patch's mesh-generation logic.
- Concatenates vertices, faces, and colors.
- Offsets face indices for each patch.

### `render(patches, camera, resolution)`

What it does:

- Renders patches from one camera.

Returns:

- An RGBA tensor shaped like `[H, W, 4]`.

Why it matters:

- This is the differentiable image that drives the loss.

How it works:

- Builds geometry.
- Projects vertices with the camera MVP matrix.
- Rasterizes triangles.
- Interpolates colors.
- Produces alpha from raster coverage.
- Antialiases the result.
- Flips vertically to match the UI/image coordinate convention.

### `render_both(patches, cameras)`

What it does:

- Renders both scene cameras.

Returns:

- A tuple of two RGBA tensors.

Why it matters:

- The optimizer compares each camera view to its corresponding target.

How it works:

- Calls `render()` once per camera.

## `core/loss.py`

### `_match_size(target, rendered)`

What it does:

- Resizes or formats a target tensor to match a rendered image.

Returns:

- A target tensor compatible with the rendered tensor.

Why it matters:

- Pixel losses require matching image dimensions.

How it works:

- Uses tensor interpolation when spatial sizes differ.

### `mse_loss(rendered, target, mask=None)`

What it does:

- Computes RGB mean squared error.

Returns:

- A scalar torch tensor.

Why it matters:

- It is the basic appearance-matching loss.

How it works:

- Compares rendered RGB against target RGB.
- If a mask is provided, weights the pixel error by that mask.

### `silhouette_loss(rendered, target_mask)`

What it does:

- Compares rendered alpha against a foreground mask.

Returns:

- A scalar torch tensor.

Why it matters:

- Encourages the visible patch silhouette to match the target shape.

How it works:

- Extracts alpha from rendered RGBA.
- Computes MSE against the target mask.

### `masked_rgb_loss(rendered, target, target_mask)`

What it does:

- Computes RGB loss focused on foreground pixels.

Returns:

- A scalar torch tensor.

Why it matters:

- Avoids letting the background dominate the image loss.

How it works:

- Computes squared RGB error.
- Weights the error by the foreground mask.
- Normalizes by mask area.

## `core/optimizer.py`

### `parse_palette(text)`

What it does:

- Parses user-entered colors.

Returns:

- A tuple of RGB color tuples in `[0, 1]`.

Why it matters:

- The renderer and patch colors use normalized RGB values.

How it works:

- Reads comma-separated hex colors.
- Converts each `#RRGGBB` value into normalized floats.
- Falls back to `DEFAULT_PALETTE` if parsing fails.

### `image_to_tensor(image, device)`

What it does:

- Converts an image array into a torch tensor.

Returns:

- A float tensor in `[0, 1]`.

Why it matters:

- Loss functions operate on torch tensors.

How it works:

- Converts grayscale to RGB if needed.
- Drops alpha if present.
- Converts `uint8` values to float RGB.

### `fit_image_to_resolution(image, resolution, device)`

What it does:

- Fits a target image into the renderer resolution without stretching.

Returns:

- A tensor shaped like `[H, W, 3]`.

Why it matters:

- Prevents target images from being squashed to match the camera render shape.

How it works:

- Computes a uniform scale that preserves aspect ratio.
- Resizes the image.
- Pads the remaining area with an estimated background color.

### `quantize_to_palette(rgb, palette_tensor)`

What it does:

- Assigns a color to the nearest palette color.

Returns:

- A quantized RGB tensor.

Why it matters:

- Keeps rendered patch colors tied to the fabrication palette.

How it works:

- Computes distance from the input color to each palette color.
- Selects the closest palette color.

### `foreground_mask_from_image(image, palette, device, resolution=None)`

What it does:

- Creates a foreground mask from a target image.

Returns:

- A float mask tensor shaped like `[H, W, 1]`.

Why it matters:

- Silhouette and masked RGB losses need to know what is target foreground.

How it works:

- Fits the image to the renderer resolution if requested.
- Estimates background color from image corners.
- Marks pixels that differ from the background as foreground.

### `snap_patches_to_palette(patches, palette)`

What it does:

- Forces patch albedos to palette colors.

Returns:

- Nothing.

Why it matters:

- Keeps viewport and renderer colors consistent with the user palette.

How it works:

- For each patch, finds the nearest palette color to the current albedo.
- Replaces the patch albedo with that palette color.

### `_parameter_groups(patches)`

What it does:

- Collects differentiable parameters for Adam.

Returns:

- A list of tensors.

Why it matters:

- Controls exactly what optimization can change.

How it works:

- Includes patch center and theta.
- Includes each control point's x, y, handle scale, and handle rotation.
- Freezes local control point z so patches stay flat in local space.

### `_patch_collision_radius(patch)`

What it does:

- Estimates a conservative collision radius for a patch.

Returns:

- A scalar tensor.

Why it matters:

- The overlap penalty needs an approximate patch size.

How it works:

- Samples local outline points.
- Measures distance from the local center.
- Uses the maximum distance as the radius.

### `patch_overlap_loss(patches, margin)`

What it does:

- Penalizes patches whose approximate bounding circles overlap.

Returns:

- A scalar torch tensor.

Why it matters:

- Discourages pieces from intersecting without using a hard separation constraint.

How it works:

- Computes pairwise center distances.
- Computes allowed distance from patch radii plus margin.
- Penalizes positive overlap with squared error.

### `_patch_projected_area(patch, camera, device)`

What it does:

- Estimates how much screen area a patch covers in one camera.

Returns:

- A scalar tensor.

Why it matters:

- Used by the per-piece visibility penalty.

How it works:

- Projects patch outline points into normalized device coordinates.
- Computes polygon area with the shoelace formula.

### `patch_visibility_loss(patches, cameras, min_area)`

What it does:

- Penalizes pieces that are too invisible in camera views.

Returns:

- A scalar torch tensor.

Why it matters:

- Helps avoid degenerate solutions where a patch turns edge-on or disappears from the important views.

How it works:

- Computes projected area for each patch in each camera.
- Penalizes area below `min_area`.

### `constrain_theta_to_camera_band(theta, camera_angles, margin)`

What it does:

- Projects a patch rotation into the allowed angular bands.

Returns:

- A Python float theta value.

Why it matters:

- This replaces the old edge-on penalty with a hard constraint. A patch cannot remain within 15 degrees of either camera yaw, which prevents edge-on degenerate orientations without adding another soft loss term.

How it works:

- Wraps theta into a half-turn range because opposite patch normals represent the same flat piece orientation.
- Computes camera yaw angles from camera position relative to target.
- If theta is already at least `margin` away from every camera yaw, it is kept.
- Otherwise, theta is moved to the nearest valid boundary.
- With cameras at 0 and 90 degrees and a 15 degree margin, the allowed bands are `[-75, -15]` and `[15, 75]` degrees.

### `SceneOptimizer.__init__(...)`

What it does:

- Prepares the differentiable optimization problem.

Returns:

- A `SceneOptimizer`.

Why it matters:

- This object owns renderer state, target tensors, masks, optimizer, palette, and penalty weights.

How it works:

- Stores patches and cameras.
- Parses palette.
- Converts targets to tensors.
- Fits target images to renderer resolution.
- Builds foreground masks.
- Creates a `DiffRenderer`.
- Builds Adam parameter groups.
- Snaps initial colors to the palette.
- Applies initial post-step constraints.
- Creates a `StochasticRewriteDescent` instance with the selected enable state and patch count penalty.

### `SceneOptimizer.step(step_idx=1, total_steps=1)`

What it does:

- Runs one optimization iteration.

Returns:

- A metrics dictionary with values such as total loss, view losses, overlap penalty, visibility penalty, camera-bounds penalty, patch count penalty, weighted geometric penalty contributions, and SRD add/delete totals.

Why it matters:

- This is the core training step.

How it works:

- Clears gradients.
- Renders view 1 and view 2.
- Computes RGB and silhouette losses.
- Computes negative-space losses for both target-image views.
- Computes overlap, visibility, and camera-bounds penalties.
- Computes `lambda_count * num_active_patches`; this is constant for Adam but matters when SRD compares structural rewrites.
- Combines all terms into one scalar total loss.
- Calls `backward()`.
- Calls Adam `step()`.
- Clears gradients.
- Applies post-step constraints.
- Enforces the camera-angle theta constraint during post-step constraints.
- Calls `srd.step(...)` after the Adam step when SRD is enabled.

### `SceneOptimizer._loss_from_renders(render1, render2, patches)`

What it does:

- Computes the full scalar optimization loss from already-rendered camera images.

Returns:

- A tuple of `(loss, components)`, where `loss` is a scalar tensor and `components` is a dictionary of tensor loss parts.

Why it matters:

- The main training step needs one shared loss definition for rendered images and geometric penalties.

How it works:

- Computes view 1 masked RGB and silhouette losses.
- Computes view 1 negative-space loss from rendered alpha in target background pixels.
- Computes view 2 image loss.
- Computes view 2 negative-space loss when a second target image is available.
- Computes overlap, visibility, and camera-bounds penalties for the supplied patch sequence.
- Combines the weighted terms into the same total loss used for optimization.

### `SceneOptimizer._post_step_constraints()`

What it does:

- Enforces hard cleanup constraints after each step.

Returns:

- Nothing.

Why it matters:

- Keeps optimization from producing invalid or non-flat patch parameters.

How it works:

- Replaces NaNs in centers, rotations, and control parameters.
- Forces each control point's local z value to zero.
- Clamps handle scale to a valid range.
- Snaps albedo back to the palette.

### `SceneOptimizer.mesh_snapshot()`

What it does:

- Converts current patches to detached viewport meshes.

Returns:

- A list of `Mesh` objects.

Why it matters:

- The worker can send mesh snapshots to the UI without exposing live autograd tensors.

How it works:

- Calls `patch.to_mesh()` for each patch.

### `SceneOptimizer.run(n_steps, callback=None)`

What it does:

- Runs a fixed number of optimization steps.

Returns:

- The optimized patch list.

Why it matters:

- Provides a simple non-UI optimization loop.

How it works:

- Calls `step()` repeatedly.
- Invokes the callback after each step if provided.

## `ui/image_panel.py`

### `ImageDropZone`

What it does:

- Provides drag-and-drop and click-to-load image behavior.

Returns:

- A widget instance.

Why it matters:

- This is how target images enter the app.

How it works:

- Accepts image file drops.
- Opens a file picker on click.
- Emits `image_loaded`.
- Displays the loaded image scaled while preserving aspect ratio.

### `ImagePanel.set_camera_previews(meshes, cameras)`

What it does:

- Updates the two camera preview widgets.

Returns:

- Nothing.

Why it matters:

- Shows what the scene cameras currently see under the current patch layout.

How it works:

- Calls `_render_camera_preview()` for each camera.
- Sends the resulting pixmaps to the preview widgets.

### `_render_camera_preview(meshes, camera, width, height)`

What it does:

- Renders a simple software preview from one camera.

Returns:

- A `QPixmap`.

Why it matters:

- Gives immediate UI feedback without needing to read pixels from the differentiable renderer.

How it works:

- Projects mesh vertices through camera view/projection matrices.
- Converts normalized device coordinates into image pixels.
- Draws triangle polygons with `QPainter`.

## `ui/viewport.py`

### `Viewport.set_patches(patches)`

What it does:

- Displays a list of patch objects in the viewport.

Returns:

- Nothing.

Why it matters:

- Shows initialized patch geometry before optimization starts.

How it works:

- Converts each patch to a `Mesh`.
- Stores the resulting meshes in the scene.
- Triggers a repaint.

### `Viewport.set_meshes(meshes)`

What it does:

- Replaces the visible scene meshes.

Returns:

- Nothing.

Why it matters:

- Used during optimization updates.

How it works:

- Replaces scene mesh data.
- Triggers a repaint.

### `Viewport.reset()`

What it does:

- Clears the viewport to its empty state.

Returns:

- Nothing.

Why it matters:

- Supports the reset button.

How it works:

- Clears scene meshes.
- Resets viewport camera framing.
- Triggers a repaint.

### OpenGL paint helpers

What they do:

- Draw grid, axes, camera frustums, and meshes.

Returns:

- Nothing.

Why they matter:

- They provide the visual editing context.

How they work:

- Upload vertex/color data to OpenGL buffers.
- Use shader programs for colored line and triangle drawing.

## Loss Functions

The optimizer combines image-space losses with geometric penalties.

### View 1 image loss

View 1 always uses target-image matching:

```text
view1_rgb = masked_rgb_loss(render1, target1, mask1)
view1_total = view1_rgb
            + silhouette_weight * silhouette_loss(render1, mask1)
            + negative_space_weight * negative_space_loss(render1, mask1)
```

Significance:

- The RGB term matches foreground color.
- The silhouette term matches the target shape using rendered alpha.
- The total term is the full per-view contribution used by the optimizer.

### View 2 image loss

If a second target image exists:

```text
view2_rgb = masked_rgb_loss(render2, target2, mask2)
view2_total = view2_rgb
            + silhouette_weight * silhouette_loss(render2, mask2)
            + negative_space_weight * negative_space_loss(render2, mask2)
```

### Silhouette loss

```text
silhouette_loss = mean((rendered_alpha - target_foreground_mask)^2)
```

Significance:

- Encourages the patch union seen by a camera to match the target silhouette.

### Masked RGB loss

```text
masked_rgb_loss = sum(mask * (rendered_rgb - target_rgb)^2) / sum(mask)
```

Significance:

- Focuses color matching on target foreground instead of wasting loss on padded or background pixels.

### Negative-space loss

```text
negative_space_loss = sum((rendered_alpha^2) * (1 - target_foreground_mask))
                    / sum(1 - target_foreground_mask)
```

Significance:

- Penalizes patches that appear in the target image's background region.
- This is computed separately for view 1 and view 2.
- The default `negative_space_weight` is `3.0`, so false positives outside the silhouette are weighted more strongly than the foreground RGB matching term.

### Patch count penalty

```text
patch_count_loss = lambda_count * num_active_patches
```

Significance:

- Adds pressure for SRD to prefer simpler structures with fewer pieces.
- The default UI value is `0.05`.
- This term is discrete with respect to patch structure, so Adam cannot reduce it directly; SRD uses it when accepting or rejecting add, split, restore, and delete rewrites.

### Soft overlap penalty

```text
overlap_ij = relu(radius_i + radius_j + margin - distance(center_i, center_j))
overlap_loss = mean(overlap_ij^2)
```

Significance:

- Discourages pieces from intersecting, but does not force them apart as aggressively as a hard constraint.

### Per-piece camera visibility penalty

```text
visibility_loss = mean(relu(min_projected_area - projected_area(piece, camera))^2)
```

Significance:

- Prevents patches from disappearing by becoming too small or nearly invisible from the cameras.

### Total loss

The main optimizer step combines terms as:

```text
total_loss =
    view1_total
  + view2_total
  + overlap_weight * overlap_loss
  + visibility_weight * visibility_loss
  + patch_count_loss
```

Current important weights are set in `SceneOptimizer.__init__()`:

- `silhouette_weight`
- `negative_space_weight`
- `overlap_weight`
- `visibility_weight`
- `min_projected_area`
- `overlap_margin`
- `lambda_count`, controlled by the SRD patch count penalty slider.

The edge-on behavior is handled as a hard theta constraint, not a loss term.

## Optimization Methods

### Adam

The main optimizer is `torch.optim.Adam`.

Optimized parameters include:

- Patch center.
- Patch theta.
- Control point x.
- Control point y.
- Control point handle scale.
- Control point handle rotation.

Local control point z is frozen and reset to zero after each step so each patch stays flat in its local coordinates.

### Differentiable rendering

The rendered images are produced by `nvdiffrast`.

Why it matters:

- Gradients can flow from image loss back to patch positions, rotations, shape parameters, and handles.

### Palette snapping

Patch albedos are snapped to the nearest user palette color.

Why it matters:

- The viewport, previews, and rendered optimization state stay tied to fabrication-ready colors.

### Hard post-step constraints

After every optimizer step, `_post_step_constraints()`:

- Removes NaNs.
- Constrains patch theta to stay at least 15 degrees away from each camera yaw.
- Forces local control point z values to zero.
- Clamps handle scale.
- Snaps albedo to the palette.

Why it matters:

- Adam can produce invalid values if gradients become unstable. This pass keeps the patch parameterization usable.

### Fixed steps versus convergence mode

The worker supports two run modes:

- Fixed steps: run for the selected number of steps and update a progress bar.
- Until convergence: keep running until `metrics["loss"] <= convergence_threshold` or the user stops the worker.

Why it matters:

- Fixed steps are predictable and easy to compare.
- Convergence mode is useful when the user wants the tool to keep improving until the loss is small enough.
