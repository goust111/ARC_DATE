import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

get_bg = _ARC_UTILS.get_bg
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste


POSES = (
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
)
IDENTITY_POSE = (False, 0)
LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
SEPARATOR = 1
MAX_GEN_ATTEMPTS = 400
PANEL_ATTEMPTS = 300
DISTRACTOR_ATTEMPTS = 200
ALL_COLORS = tuple(range(10))


def _apply_pose(panel: np.ndarray, pose: tuple[bool, int]) -> np.ndarray:
    mirrored, turns = pose
    out = np.fliplr(panel) if mirrored else np.array(panel, copy=True)
    if turns:
        out = np.rot90(out, int(turns))
    return np.array(out, copy=True)


def _build_inverse_poses() -> dict[tuple[bool, int], tuple[bool, int]]:
    probe = np.arange(9, dtype=int).reshape(3, 3)
    inverse = {}
    for pose in POSES:
        for cand in POSES:
            if np.array_equal(_apply_pose(_apply_pose(probe, cand), pose), probe):
                inverse[pose] = cand
                break
        assert pose in inverse
    return inverse


INVERSE_POSE = _build_inverse_poses()


def _sparsify(panel: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(panel.shape, int(bg), dtype=int)
    out[::2, ::2] = panel[::2, ::2]
    return out


def _non_separator_spans(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, is_sep in enumerate(separator_mask):
        if not is_sep and start is None:
            start = i
        elif is_sep and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(separator_mask) - 1))
    return spans


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    bg = int(get_bg(grid))
    row_spans = _non_separator_spans(np.all(grid == bg, axis=1))
    col_spans = _non_separator_spans(np.all(grid == bg, axis=0))
    assert row_spans and col_spans, "expected a separator-based panel layout"

    heights = {int(r1 - r0 + 1) for r0, r1 in row_spans}
    widths = {int(c1 - c0 + 1) for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "panels must have equal size"
    panel_size = next(iter(heights))
    assert panel_size == next(iter(widths)) and panel_size % 2 == 1, "panels must be odd square boards"

    panels = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panels.append(crop_bbox(grid, (r0, c0, r1, c1)))
    return panels, int(panel_size)


def _is_dot_panel(panel: np.ndarray, bg: int) -> bool:
    off_sample = np.ones(panel.shape, dtype=bool)
    off_sample[::2, ::2] = False
    return bool(np.all(panel[off_sample] == int(bg)) and np.any(panel[::2, ::2] != int(bg)))


def _matching_poses(panel: np.ndarray, dot_panel: np.ndarray, bg: int) -> list[tuple[bool, int]]:
    matches = []
    for pose in POSES:
        transformed = _apply_pose(panel, pose)
        if np.array_equal(_sparsify(transformed, bg), dot_panel):
            matches.append(pose)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the dense pattern whose transformed even-even subsample matches the sparse dot board.

    Input structure:
    - The grid is partitioned by full-background separator bands into equal odd square panels.
    - Exactly one panel is a sparse dot lattice: only even-row/even-column cells may be colored, and all other cells are background.
    - The remaining panels are dense multicolor candidate boards shown in their original poses.

    Rule:
    - Find the unique candidate and unique pose among the 8 allowed poses: rotate by 0/90/180/270 degrees, or mirror left-right first and then rotate.
    - For each candidate pose, compare the whole sparse projection obtained by keeping only even-row/even-column cells and setting every other cell to background.
    - Exactly one posed candidate matches the dot panel at every sampled position, including sampled background cells.
    - Output that candidate in the matching pose as a full dense board.

    Key insight:
    - The sparse panel is not the answer itself; it is a downsampled projection that jointly reveals both which candidate to use and which pose to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    panels, _panel_size = _extract_panels(g)
    dot_indices = [i for i, panel in enumerate(panels) if _is_dot_panel(panel, bg)]
    assert len(dot_indices) == 1, "expected exactly one sparse dot panel"
    dot_panel = panels[dot_indices[0]]

    matches = []
    for i, panel in enumerate(panels):
        if i == dot_indices[0]:
            continue
        for pose in POSES:
            transformed = _apply_pose(panel, pose)
            if np.array_equal(_sparsify(transformed, bg), dot_panel):
                matches.append((i, pose, transformed))
    assert len(matches) == 1, "candidate-pose match must be unique"
    return matches[0][2]


def _panel_color_counts(panel: np.ndarray, bg: int) -> list[int]:
    return [int(np.count_nonzero(panel[r] != bg)) for r in range(panel.shape[0])] + [
        int(np.count_nonzero(panel[:, c] != bg)) for c in range(panel.shape[1])
    ]


def _panel_is_dense_candidate(panel: np.ndarray, bg: int) -> bool:
    if _is_dot_panel(panel, bg):
        return False
    counts = _panel_color_counts(panel, bg)
    if min(counts) < 2:
        return False
    if np.count_nonzero(panel != bg) <= panel.size // 2:
        return False
    colors = sorted(set(map(int, np.unique(panel))) - {int(bg)})
    if len(colors) < 3:
        return False
    off_sample = np.ones(panel.shape, dtype=bool)
    off_sample[::2, ::2] = False
    if not bool(np.any(panel[off_sample] != int(bg))):
        return False
    return True


def _pose_group_choice(rng: random.Random) -> tuple[bool, int]:
    if rng.random() < 0.5:
        return False, rng.choice((1, 2, 3))
    return True, rng.choice((0, 1, 2, 3))


def _repair_line_density(panel: np.ndarray, bg: int, palette: list[int], rng: random.Random) -> None:
    for r in range(panel.shape[0]):
        while int(np.count_nonzero(panel[r] != bg)) < 2:
            c = rng.randrange(panel.shape[1])
            panel[r, c] = rng.choice(palette)
    for c in range(panel.shape[1]):
        while int(np.count_nonzero(panel[:, c] != bg)) < 2:
            r = rng.randrange(panel.shape[0])
            panel[r, c] = rng.choice(palette)


def _random_dense_panel(size: int, bg: int, palette: list[int], rng: random.Random) -> np.ndarray | None:
    sample_min = max(3, (size // 2))
    for _ in range(PANEL_ATTEMPTS):
        panel = np.full((size, size), int(bg), dtype=int)
        for r in range(size):
            for c in range(size):
                fill_prob = 0.52 if (r % 2 == 0 and c % 2 == 0) else 0.83
                if rng.random() < fill_prob:
                    panel[r, c] = rng.choice(palette)
        _repair_line_density(panel, bg, palette, rng)

        sample = panel[::2, ::2]
        sample_fg = int(np.count_nonzero(sample != bg))
        sample_bg = int(np.count_nonzero(sample == bg))
        if sample_fg < sample_min or sample_bg < 1:
            continue
        colors = sorted(set(map(int, np.unique(panel))) - {int(bg)})
        if len(colors) < min(3, len(palette)):
            continue
        pose_signatures = {(_apply_pose(panel, pose)).tobytes() for pose in POSES}
        if len(pose_signatures) < len(POSES):
            continue
        if not _panel_is_dense_candidate(panel, bg):
            continue
        return panel
    return None


def _pick_palette(bg: int, rng: random.Random) -> list[int]:
    colors = [c for c in ALL_COLORS if c != bg]
    rng.shuffle(colors)
    return colors[: rng.randint(3, 5)]


def _make_structured_distractor(
    solved: np.ndarray,
    dot_panel: np.ndarray,
    bg: int,
    palette: list[int],
    rng: random.Random,
    mode: str,
) -> np.ndarray | None:
    sample_bg_cells = [(r, c) for r in range(0, solved.shape[0], 2) for c in range(0, solved.shape[1], 2) if int(solved[r, c]) == bg]
    sample_fg_cells = [(r, c) for r in range(0, solved.shape[0], 2) for c in range(0, solved.shape[1], 2) if int(solved[r, c]) != bg]
    if mode == "extra_sample_fg" and not sample_bg_cells:
        return None
    if mode == "wrong_sample_color" and not sample_fg_cells:
        return None

    for _ in range(DISTRACTOR_ATTEMPTS):
        near = np.array(solved, copy=True)
        if mode == "extra_sample_fg":
            r, c = rng.choice(sample_bg_cells)
            near[r, c] = rng.choice(palette)
        elif mode == "wrong_sample_color":
            r, c = rng.choice(sample_fg_cells)
            choices = [color for color in palette if color != int(near[r, c])]
            near[r, c] = rng.choice(choices)
        else:
            return None

        pose = rng.choice(POSES)
        base = _apply_pose(near, INVERSE_POSE[pose])
        if not _panel_is_dense_candidate(base, bg):
            continue
        if _matching_poses(base, dot_panel, bg):
            continue
        return base
    return None


def _make_random_distractor(size: int, dot_panel: np.ndarray, bg: int, palette: list[int], rng: random.Random) -> np.ndarray | None:
    for _ in range(DISTRACTOR_ATTEMPTS):
        panel = _random_dense_panel(size, bg, palette, rng)
        if panel is None:
            return None
        if _matching_poses(panel, dot_panel, bg):
            continue
        return panel
    return None


def _assemble_input(
    rows: int,
    cols: int,
    dot_index: int,
    dot_panel: np.ndarray,
    candidates: list[np.ndarray],
    bg: int,
) -> np.ndarray:
    size = dot_panel.shape[0]
    grid = np.full((rows * size + (rows - 1) * SEPARATOR, cols * size + (cols - 1) * SEPARATOR), int(bg), dtype=int)
    panels = list(candidates)
    panels.insert(dot_index, dot_panel)
    for idx, panel in enumerate(panels):
        rr = idx // cols
        cc = idx % cols
        r0 = rr * (size + SEPARATOR)
        c0 = cc * (size + SEPARATOR)
        paste(grid, panel, r0, c0)
    return grid


def _validate_instance(
    inp: np.ndarray,
    solved: np.ndarray,
    true_candidate: np.ndarray,
    target_pose: tuple[bool, int],
) -> bool:
    bg = int(get_bg(inp))
    if inp.ndim != 2 or max(inp.shape) > 30 or max(solved.shape) > 30:
        return False
    if int(np.count_nonzero(inp == bg)) <= max(int(np.count_nonzero(inp == color)) for color in range(10) if color != bg):
        return False

    try:
        panels, _panel_size = _extract_panels(inp)
    except AssertionError:
        return False
    dot_indices = [i for i, panel in enumerate(panels) if _is_dot_panel(panel, bg)]
    if len(dot_indices) != 1:
        return False
    dot_panel = panels[dot_indices[0]]
    candidate_panels = [panel for i, panel in enumerate(panels) if i != dot_indices[0]]
    if not all(_panel_is_dense_candidate(panel, bg) for panel in candidate_panels):
        return False

    true_pose_matches = _matching_poses(true_candidate, dot_panel, bg)
    if true_pose_matches != [target_pose]:
        return False
    if np.array_equal(_sparsify(true_candidate, bg), dot_panel):
        return False

    hits = []
    for panel in candidate_panels:
        for pose in _matching_poses(panel, dot_panel, bg):
            hits.append((_apply_pose(panel, pose)).tobytes())
    if len(hits) != 1:
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(out, solved))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUT_CHOICES)
    sizes = [size for size in range(5, 30, 2) if rows * size + (rows - 1) * SEPARATOR <= 30 and cols * size + (cols - 1) * SEPARATOR <= 30]
    size = rng.choice(sizes)
    bg = rng.choice(ALL_COLORS)
    palette = _pick_palette(bg, rng)
    solved = _random_dense_panel(size, bg, palette, rng)
    if solved is None:
        return None

    target_pose = _pose_group_choice(rng)
    true_candidate = _apply_pose(solved, INVERSE_POSE[target_pose])
    dot_panel = _sparsify(solved, bg)
    if _matching_poses(true_candidate, dot_panel, bg) != [target_pose]:
        return None
    if np.array_equal(_sparsify(true_candidate, bg), dot_panel):
        return None

    candidate_count = rows * cols - 1
    candidates = [true_candidate]
    for mode in ("extra_sample_fg", "wrong_sample_color"):
        distractor = _make_structured_distractor(solved, dot_panel, bg, palette, rng, mode)
        if distractor is None:
            return None
        candidates.append(distractor)
    while len(candidates) < candidate_count:
        distractor = _make_random_distractor(size, dot_panel, bg, palette, rng)
        if distractor is None:
            return None
        if any(np.array_equal(distractor, existing) for existing in candidates):
            continue
        candidates.append(distractor)

    rng.shuffle(candidates)
    dot_index = rng.randrange(rows * cols)
    inp = _assemble_input(rows, cols, dot_index, dot_panel, candidates, bg)
    if not _validate_instance(inp, solved, true_candidate, target_pose):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-subsample reconstruction input")
