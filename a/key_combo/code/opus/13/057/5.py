from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
FRAME = 5
TEAL = 8

SOURCE_INNER_HEIGHT = 4
SOURCE_VISIBLE_ROWS = 2
TARGET_COUNT_RANGE = (3, 5)
TEMPLATE_WIDTH_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_bboxes(mask: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = (int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max()))
            out.append((arr, bbox))
    return out


def _is_rectangular_frame(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(coords) != expected:
        return False
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    perimeter = np.zeros_like(mask)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    return bool(np.array_equal(mask, perimeter))


def _find_panels(grid: np.ndarray) -> list[dict]:
    panels: list[dict] = []
    for coords, bbox in _component_bboxes(grid == FRAME):
        if not _is_rectangular_frame(coords, bbox):
            continue
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        panels.append(
            {
                "bbox": bbox,
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "inner": inner,
            }
        )
    return panels


def _is_source_inner(inner: np.ndarray) -> bool:
    if inner.shape[0] != SOURCE_INNER_HEIGHT:
        return False
    if not np.all(inner[:SOURCE_VISIBLE_ROWS] == BG):
        return False
    lower_colors = set(map(int, np.unique(inner[SOURCE_VISIBLE_ROWS:])))
    if not lower_colors.issubset({BG, BLUE, TEAL}):
        return False
    return bool(np.any(inner[SOURCE_VISIBLE_ROWS:] != BG))


def _build_template(source_inner: np.ndarray) -> np.ndarray:
    lower = source_inner[SOURCE_VISIBLE_ROWS:].copy()
    return np.vstack([lower[::-1], lower])


def _transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    panels = _find_panels(g)
    assert panels, "expected gray framed panels"

    source_candidates = [panel for panel in panels if _is_source_inner(panel["inner"])]
    assert len(source_candidates) == 1, "expected exactly one template source panel"
    source_panel = source_candidates[0]
    source_shape = tuple(int(v) for v in source_panel["inner"].shape)

    targets = [panel for panel in panels if panel is not source_panel]
    assert len(targets) >= 1, "expected at least one target panel"
    assert all(tuple(int(v) for v in panel["inner"].shape) == source_shape for panel in targets)

    template = _build_template(source_panel["inner"])
    mask = template != BG
    out = g.copy()
    stamped: list[tuple[int, int]] = []
    counts: list[int] = []

    for panel in targets:
        r0, c0, r1, c1 = panel["inner_bbox"]
        inner = panel["inner"]
        red_count = int(np.count_nonzero(inner == RED))
        counts.append(red_count)
        if red_count != 2:
            continue
        stamped.append((r0, c0))
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        original = region.copy()
        region[mask] = template[mask]
        region[(original == RED) & ~mask] = BG

    info = {
        "panel_count": len(panels),
        "target_count": len(targets),
        "red_counts": counts,
        "template": template,
        "template_mask": mask,
        "source_bbox": source_panel["bbox"],
        "stamped_origins": stamped,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete the half-filled template and stamp it only into target panels with exactly two red markers.

    Input structure:
    - A black background contains several disjoint gray rectangular frames with 1-cell-thick borders.
    - Exactly one frame is the template source: its 4xW interior has a blank top half and a blue/teal pattern only in the bottom half.
    - The other same-sized framed interiors are target panels containing green pixels plus 0-3 red marker pixels.

    Rule:
    - Reconstruct the full 4xW template by copying the source panel's bottom two rows into the top two rows in reversed row order.
    - Count red markers inside each target panel.
    - For every target whose red-marker count is exactly 2, transparently overlay the completed template onto that target, with black template cells left transparent.
    - In stamped targets, any original red marker not covered by a non-black template cell is erased to black; all other panels and all gray borders stay unchanged.

    Key insight:
    - The visible half-pattern is only an intermediate clue: it must first be mirrored into a full stamp, and that stamp is then applied selectively using the red-marker count as the gating signal.
    """
    return _transform(grid)[0]


def _sample_connected_mask(height: int, width: int, size_range: tuple[int, int], *, require_both_rows: bool) -> np.ndarray | None:
    area = height * width
    min_size, max_size = size_range
    max_size = min(max_size, area - 1)
    if min_size > max_size:
        return None
    for _ in range(80):
        target_size = random.randint(min_size, max_size)
        mask = np.zeros((height, width), dtype=bool)
        seed = (random.randrange(height), random.randrange(width))
        mask[seed] = True
        frontier = {seed}
        while np.count_nonzero(mask) < target_size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            rr, cc = cell
            neighbors = []
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            random.shuffle(neighbors)
            grew = False
            for nr, nc in neighbors:
                if np.count_nonzero(mask) >= target_size:
                    break
                mask[nr, nc] = True
                frontier.add((nr, nc))
                grew = True
            if grew:
                frontier.add(cell)
        if np.count_nonzero(mask) != target_size:
            continue
        if require_both_rows and (not mask[0].any() or not mask[-1].any()):
            continue
        if np.all(mask):
            continue
        return mask
    return None


def _make_source_inner(width: int) -> np.ndarray | None:
    lower_mask = _sample_connected_mask(2, width, (3, 6), require_both_rows=True)
    if lower_mask is None:
        return None

    lower = np.zeros((2, width), dtype=int)
    fill_positions = list(zip(*np.where(lower_mask)))
    if len(fill_positions) >= 2 and random.random() < 0.65:
        teal_count = random.randint(1, len(fill_positions) - 1)
        teal_cells = set(random.sample(fill_positions, teal_count))
    else:
        teal_cells = set()
    for pos in fill_positions:
        lower[pos] = TEAL if pos in teal_cells else BLUE

    inner = np.zeros((SOURCE_INNER_HEIGHT, width), dtype=int)
    inner[SOURCE_VISIBLE_ROWS:] = lower
    template = _build_template(inner)
    if not np.any(template == BG) or not np.any(template != BG):
        return None
    return inner


def _place_connected_green(inner: np.ndarray, blocked: set[tuple[int, int]], count_range: tuple[int, int]) -> bool:
    height, width = inner.shape
    available = [(r, c) for r in range(height) for c in range(width) if (r, c) not in blocked]
    if len(available) < count_range[0]:
        return False
    for _ in range(80):
        target_size = random.randint(count_range[0], min(count_range[1], len(available)))
        seed = random.choice(available)
        cells = {seed}
        frontier = {seed}
        while len(cells) < target_size and frontier:
            rr, cc = random.choice(tuple(frontier))
            choices = []
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in blocked and (nr, nc) not in cells:
                    choices.append((nr, nc))
            if not choices:
                frontier.discard((rr, cc))
                continue
            nxt = random.choice(choices)
            cells.add(nxt)
            frontier.add(nxt)
        if len(cells) != target_size:
            continue
        for r, c in cells:
            inner[r, c] = GREEN
        return True
    return False


def _make_target_inner(width: int, template_mask: np.ndarray, red_count: int, role: str) -> np.ndarray | None:
    inner = np.zeros((SOURCE_INNER_HEIGHT, width), dtype=int)
    filled_positions = list(zip(*np.where(template_mask)))
    empty_positions = list(zip(*np.where(~template_mask)))
    red_positions: list[tuple[int, int]]

    if role == "showcase":
        if not filled_positions or not empty_positions:
            return None
        red_positions = [random.choice(filled_positions), random.choice(empty_positions)]
    else:
        pool = filled_positions + empty_positions
        if red_count > len(pool):
            return None
        red_positions = random.sample(pool, red_count)

    red_set = set(red_positions)
    for r, c in red_positions:
        inner[r, c] = RED

    if role == "showcase":
        green_min, green_max = 3, 5
    else:
        green_min, green_max = 2, 6
    if not _place_connected_green(inner, red_set, (green_min, green_max)):
        return None
    return inner


def _paste_panel(canvas: np.ndarray, inner: np.ndarray, top: int, left: int) -> None:
    h, w = inner.shape
    outer_h, outer_w = h + 2, w + 2
    canvas[top : top + outer_h, left : left + outer_w] = BG
    canvas[top, left : left + outer_w] = FRAME
    canvas[top + outer_h - 1, left : left + outer_w] = FRAME
    canvas[top : top + outer_h, left] = FRAME
    canvas[top : top + outer_h, left + outer_w - 1] = FRAME
    canvas[top + 1 : top + 1 + h, left + 1 : left + 1 + w] = inner


def _build_counts(target_count: int) -> list[int]:
    choices = [0, 1, 2, 3]
    for _ in range(200):
        counts = [2, random.choice([0, 1, 3])]
        while len(counts) < target_count:
            counts.append(random.choice(choices))
        random.shuffle(counts)
        if 2 not in counts:
            continue
        if all(count == 2 for count in counts):
            continue
        if len(set(counts)) < 2:
            continue
        return counts
    raise RuntimeError("failed to build target counts")


def _expected_from_construction(grid: np.ndarray, template_mask: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, dict]:
    out, info = _transform(grid)
    assert np.array_equal(info["template_mask"], template_mask)
    assert np.array_equal(info["template"], template)
    return out, info


def _gen_attempt() -> np.ndarray | None:
    width = random.choice(TEMPLATE_WIDTH_CHOICES)
    source_inner = _make_source_inner(width)
    if source_inner is None:
        return None
    template = _build_template(source_inner)
    template_mask = template != BG

    target_count = random.randint(*TARGET_COUNT_RANGE)
    counts = _build_counts(target_count)
    showcase_index = random.choice([i for i, count in enumerate(counts) if count == 2])

    inners: list[np.ndarray] = [source_inner]
    for i, count in enumerate(counts):
        role = "showcase" if i == showcase_index else "ordinary"
        inner = _make_target_inner(width, template_mask, count, role)
        if inner is None:
            return None
        inners.append(inner)

    outer_h = SOURCE_INNER_HEIGHT + 2
    outer_w = width + 2
    panel_count = len(inners)
    cols = 3 if panel_count >= 5 and random.random() < 0.75 else 2
    rows = math.ceil(panel_count / cols)
    if cols * rows < panel_count:
        return None

    gap_r = random.randint(2, 4)
    gap_c = random.randint(2, 4)
    jitter_r = 1 if gap_r >= 3 else 0
    jitter_c = 1 if gap_c >= 3 else 0

    positions: list[tuple[int, int]] = []
    for idx in range(panel_count):
        row = idx // cols
        col = idx % cols
        base_r = 1 + row * (outer_h + gap_r)
        base_c = 1 + col * (outer_w + gap_c)
        if idx == 0:
            positions.append((1, 1))
        else:
            positions.append((base_r + random.randint(0, jitter_r), base_c + random.randint(0, jitter_c)))

    height = max(r + outer_h for r, _ in positions) + 1
    width_total = max(c + outer_w for _, c in positions) + 1
    if max(height, width_total) > 30:
        return None

    canvas = np.zeros((height, width_total), dtype=int)
    for inner, (top, left) in zip(inners, positions):
        _paste_panel(canvas, inner, top, left)

    expected, info = _expected_from_construction(canvas, template_mask, template)
    red_counts = info["red_counts"]
    if red_counts != counts:
        return None
    if len(info["stamped_origins"]) != counts.count(2):
        return None
    if not any(count != 2 for count in red_counts):
        return None

    target_panels = [panel for panel in _find_panels(canvas) if panel["bbox"] != info["source_bbox"]]
    stamped_uncovered_red = False
    stamped_covered_red = False
    for panel, count in zip(target_panels, red_counts):
        inner = panel["inner"]
        if count != 2:
            continue
        if np.any((inner == RED) & ~template_mask):
            stamped_uncovered_red = True
        if np.any((inner == RED) & template_mask):
            stamped_covered_red = True
    if not stamped_uncovered_red or not stamped_covered_red:
        return None

    if not np.array_equal(expected, main(canvas)):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
