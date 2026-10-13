import numpy as np


BACKGROUND = 0
SPRITE_COLOR = 2
FRAME_COLOR = 4
CONTROL_ROWS = 3
GENERATE_ATTEMPTS = 500
_RNG = np.random.default_rng()

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_LEFT_KINDS = ("tl", "bl")
_RIGHT_KINDS = ("tr", "br")
_TEMPLATE_POOL = [
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = _DIRS4 if connectivity == 4 else _DIRS8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _triangular_columns(width: int) -> list[int]:
    cols = []
    k = 0
    while True:
        c = k * (k + 1) // 2
        if c >= width:
            break
        cols.append(int(c))
        k += 1
    return cols


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = grid == SPRITE_COLOR
    comps = _components(mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one red template"
    comp = comps[0]
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    assert c0 == 0, "template must occupy triangular column 0"
    patch = np.where(grid[r0:r1 + 1, c0:c1 + 1] == SPRITE_COLOR, SPRITE_COLOR, BACKGROUND)
    assert patch.shape in {(3, 2), (2, 3)}
    assert 3 <= int(np.count_nonzero(patch == SPRITE_COLOR)) <= 5
    assert not np.array_equal(patch, patch[:, ::-1]), "template must be horizontally asymmetric"
    return patch, box


def _frame_corners(r0: int, c0: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + 1),
        "bl": (r0 + 1, c0),
        "br": (r0 + 1, c0 + 1),
    }


def _parse_frames(grid: np.ndarray) -> list[dict[str, int | str]]:
    mask = grid == FRAME_COLOR
    comps = _components(mask, connectivity=8)
    assert len(comps) in {2, 3}, "expected 2-3 control frames"
    frames = []
    for comp in comps:
        assert len(comp) == 3, "each frame must contain exactly three yellow pixels"
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 < CONTROL_ROWS, "frames must lie in the top control band"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each frame must be a 2x2 corner marker"
        present = {tuple(map(int, cell)) for cell in comp}
        corners = _frame_corners(r0, c0)
        missing = [kind for kind, coord in corners.items() if coord not in present]
        assert len(missing) == 1
        kind = missing[0]
        side = "left" if kind in _LEFT_KINDS else "right"
        frame_col = sum(c for _, c in present) // 3
        frames.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "frame_col": int(frame_col),
                "missing_kind": kind,
                "side": side,
            }
        )
    frames.sort(key=lambda frame: int(frame["frame_col"]))
    return frames


def _nearest_frame(col: int, frames: list[dict[str, int | str]]) -> dict[str, int | str]:
    return min(frames, key=lambda frame: (abs(int(frame["frame_col"]) - int(col)), int(frame["frame_col"])))


def _stamp_with_clip(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            if patch[r, c] != SPRITE_COLOR:
                continue
            rr, cc = top + r, left + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = SPRITE_COLOR


def _render_output_from_parts(grid: np.ndarray, template: np.ndarray, stamp_top: int, frames: list[dict[str, int | str]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    flipped = template[:, ::-1]
    for col in _triangular_columns(grid.shape[1])[1:]:
        frame = _nearest_frame(col, frames)
        patch = flipped if frame["side"] == "left" else template
        _stamp_with_clip(out, patch, stamp_top, col)
    return out


def _render_constant_orientation(grid: np.ndarray, template: np.ndarray, stamp_top: int, flip: bool) -> np.ndarray:
    out = np.array(grid, copy=True)
    patch = template[:, ::-1] if flip else template
    for col in _triangular_columns(grid.shape[1])[1:]:
        _stamp_with_clip(out, patch, stamp_top, col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the left red sprite at triangular columns, flipping it by the nearest three-corner frame's missing side.

    Input structure:
    - The background is black.
    - The top three rows contain 2-3 yellow 2x2 corner frames, each missing exactly one corner.
    - The lower stamp band contains one red asymmetric sprite already placed with its bbox starting at column 0.

    Rule:
    - Crop the red sprite as the template.
    - Compute triangular-number columns 0, 1, 3, 6, 10, ... that fit in the grid width.
    - For each triangular column after 0, find the yellow frame whose average marker column is nearest, breaking ties toward the smaller column.
    - If that frame is missing a left-side corner, stamp a horizontal mirror of the template at that column; otherwise stamp the original template. Keep the frames and original template unchanged, and clip any pixels past the right edge.

    Key insight:
    - The triangular sequence decides where copies appear, but the nearest control frame decides whether each copy is mirrored, so the two spatial systems must be combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BACKGROUND
    template, (stamp_top, _c0, _r1, _c1) = _extract_template(g)
    frames = _parse_frames(g)
    tri_cols = _triangular_columns(g.shape[1])
    assert len(tri_cols) >= 5
    return _render_output_from_parts(g, template, stamp_top, frames)


def _sample_template() -> np.ndarray:
    mask = _TEMPLATE_POOL[int(_RNG.integers(len(_TEMPLATE_POOL)))]
    return np.where(mask == 1, SPRITE_COLOR, BACKGROUND)


def _sample_columns(width: int, count: int, need_tie: bool, tri_cols: list[int]) -> list[int] | None:
    if need_tie and len(tri_cols) >= 4:
        tie_choices = [c for c in tri_cols[2:-1] if 0 <= c - 2 and c + 2 <= width - 2]
        if tie_choices:
            tie_col = int(_RNG.choice(tie_choices))
            left = tie_col - 2
            right = tie_col + 2
            cols = {left, right}
            while len(cols) < count:
                candidate = int(_RNG.integers(0, width - 1))
                if all(abs(candidate - other) >= 3 for other in cols):
                    cols.add(candidate)
            return sorted(cols)
    for _ in range(80):
        cols = sorted(int(x) for x in _RNG.choice(np.arange(width - 1), size=count, replace=False))
        if all(cols[i + 1] - cols[i] >= 3 for i in range(count - 1)):
            return cols
    return None


def _sample_frames(width: int, tri_cols: list[int]) -> tuple[list[dict[str, int | str]], bool] | None:
    count = 3 if _RNG.random() < 0.65 else 2
    need_tie = bool(_RNG.random() < 0.3)
    cols = _sample_columns(width, count, need_tie, tri_cols)
    if cols is None:
        return None

    if count == 2:
        side_pattern = ["left", "right"] if _RNG.random() < 0.5 else ["right", "left"]
    else:
        side_pattern = ["left", "right", "left"] if _RNG.random() < 0.5 else ["right", "left", "right"]

    frames = []
    for c0, side in zip(cols, side_pattern):
        kind = _RNG.choice(_LEFT_KINDS if side == "left" else _RIGHT_KINDS)
        frames.append(
            {
                "r0": int(_RNG.integers(0, 2)),
                "c0": int(c0),
                "frame_col": int(c0),
                "missing_kind": str(kind),
                "side": side,
            }
        )
    return frames, need_tie


def _paint_frame(grid: np.ndarray, frame: dict[str, int | str]) -> None:
    corners = _frame_corners(int(frame["r0"]), int(frame["c0"]))
    for kind, (r, c) in corners.items():
        if kind != frame["missing_kind"]:
            grid[r, c] = FRAME_COLOR


def _build_input(width: int, template: np.ndarray, frames: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.full((7, width), BACKGROUND, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame)
    h, w = template.shape
    grid[CONTROL_ROWS:CONTROL_ROWS + h, 0:w] = np.maximum(grid[CONTROL_ROWS:CONTROL_ROWS + h, 0:w], template)
    return grid


def _semantic_checks(grid: np.ndarray, frames: list[dict[str, int | str]], template: np.ndarray, need_tie: bool) -> bool:
    tri_cols = _triangular_columns(grid.shape[1])
    cols_after_zero = tri_cols[1:]
    if len(cols_after_zero) < 4:
        return False

    assignments = [_nearest_frame(col, frames) for col in cols_after_zero]
    used = {int(frame["frame_col"]) for frame in assignments}
    if len(used) != len(frames):
        return False

    flips = [frame["side"] == "left" for frame in assignments]
    if not any(flips) or all(flips):
        return False

    switch_count = sum(flips[i] != flips[i + 1] for i in range(len(flips) - 1))
    if switch_count < 1:
        return False
    if len(frames) == 3 and switch_count < 2:
        return False

    visible_flips = [flip for col, flip in zip(cols_after_zero, flips) if col >= template.shape[1]]
    if not visible_flips or not any(visible_flips) or all(visible_flips):
        return False

    if need_tie:
        tie_cols = []
        for col in cols_after_zero:
            dists = [abs(int(frame["frame_col"]) - col) for frame in frames]
            best = min(dists)
            tied = [idx for idx, dist in enumerate(dists) if dist == best]
            if len(tied) >= 2:
                tie_cols.append(col)
        if not tie_cols:
            return False

    stamp_top = CONTROL_ROWS
    expected = _render_output_from_parts(grid, template, stamp_top, frames)
    if np.array_equal(expected, grid):
        return False
    if np.array_equal(expected, _render_constant_orientation(grid, template, stamp_top, flip=False)):
        return False
    if np.array_equal(expected, _render_constant_orientation(grid, template, stamp_top, flip=True)):
        return False
    if np.count_nonzero(expected == SPRITE_COLOR) <= np.count_nonzero(grid == SPRITE_COLOR):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = int(_RNG.integers(22, 31))
    tri_cols = _triangular_columns(width)
    if len(tri_cols) < 5:
        return None

    template = _sample_template()
    sampled = _sample_frames(width, tri_cols)
    if sampled is None:
        return None
    frames, need_tie = sampled
    grid = _build_input(width, template, frames)
    if not _semantic_checks(grid, frames, template, need_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 8")
