import random
from collections import deque

import numpy as np


_BG = 0
_GREY = 5
_LEGEND_FG = 8
_OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 9]

_CELL = 5
_ROW_COUNT = 4
_LEGEND_W = 2 * _CELL + 3
_LEGEND_H = _ROW_COUNT * _CELL + (_ROW_COUNT + 1)

_MAX_RETRIES = 300


def _mask_from_strings(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


_BASE_MASKS = {
    0: _mask_from_strings(
        [
            "###..",
            "#....",
            "####.",
            "..##.",
            "...##",
        ]
    ),
    1: _mask_from_strings(
        [
            "#####",
            "###.#",
            "#...#",
            "#.#.#",
            "#####",
        ]
    ),
    2: _mask_from_strings(
        [
            "#####",
            "#...#",
            "#####",
            "#..##",
            "#####",
        ]
    ),
    3: _mask_from_strings(
        [
            "#####",
            "###.#",
            "#.###",
            "##..#",
            "#####",
        ]
    ),
}


def _op_identity(mask: np.ndarray) -> np.ndarray:
    return mask.copy()


def _op_flip_h(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def _op_flip_v(mask: np.ndarray) -> np.ndarray:
    return np.flipud(mask)


def _op_rot90(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, -1)


def _op_rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def _op_rot270(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 1)


_OPS = {
    "id": _op_identity,
    "flip_h": _op_flip_h,
    "flip_v": _op_flip_v,
    "rot90": _op_rot90,
    "rot180": _op_rot180,
    "rot270": _op_rot270,
}
_OP_NAMES = list(_OPS)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask[:0, :0].copy()
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    mask = _crop_mask(mask.astype(bool))
    if mask.size == 0:
        return 0
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                x, y = q.popleft()
                if x in (0, h - 1) or y in (0, w - 1):
                    touches_border = True
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if not touches_border:
                holes += 1
    return holes


def _apply_op(mask: np.ndarray, op_name: str) -> np.ndarray:
    return _OPS[op_name](mask)


def _infer_op(src_mask: np.ndarray, dst_mask: np.ndarray) -> str:
    src = _crop_mask(src_mask)
    dst = _crop_mask(dst_mask)
    matches = []
    for name in _OP_NAMES:
        cand = _apply_op(src, name)
        if cand.shape == dst.shape and np.array_equal(cand, dst):
            matches.append(name)
    assert len(matches) == 1, f"expected exactly one matching transform, got {matches}"
    return matches[0]


def _legend_mapping(
    grid: np.ndarray,
) -> tuple[dict[int, str], tuple[int, int, int, int], int, int]:
    bg = _most_common_color(grid)
    line_color = None
    best_score = None
    best_bbox = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        panel = grid[r0 : r1 + 1, c0 : c1 + 1]
        full_rows = [r for r in range(panel.shape[0]) if np.all(panel[r] == color)]
        full_cols = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == color)]
        score = (len(full_rows) >= 2 and len(full_cols) >= 2, len(full_rows) + len(full_cols), pts.shape[0])
        if best_score is None or score > best_score:
            best_score = score
            line_color = color
            best_bbox = (r0, c0, r1, c1)

    assert line_color is not None and best_bbox is not None, "legend line color must exist"
    r0, c0, r1, c1 = best_bbox
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r for r in range(panel.shape[0]) if np.all(panel[r] == line_color)]
    sep_cols = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == line_color)]
    row_spans = [(a + 1, b) for a, b in zip(sep_rows, sep_rows[1:]) if a + 1 < b]
    col_spans = [(a + 1, b) for a, b in zip(sep_cols, sep_cols[1:]) if a + 1 < b]

    assert len(row_spans) >= 1 and len(col_spans) == 2, "legend grid must contain 4x2 cells"

    mapping: dict[int, str] = {}
    for rs, re in row_spans:
        left = (panel[rs:re, col_spans[0][0] : col_spans[0][1]] != bg) & (
            panel[rs:re, col_spans[0][0] : col_spans[0][1]] != line_color
        )
        right = (panel[rs:re, col_spans[1][0] : col_spans[1][1]] != bg) & (
            panel[rs:re, col_spans[1][0] : col_spans[1][1]] != line_color
        )
        holes = _hole_count(left)
        mapping[holes] = _infer_op(left, right)

    return mapping, (r0, c0, r1, c1), bg, line_color


def _extract_objects(grid: np.ndarray, main_width: int, bg: int) -> list[dict]:
    main = grid[:, :main_width]
    objects: list[dict] = []
    for color in np.unique(main):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(main == color):
            r0, c0, r1, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            global_coords = comp.copy()
            objects.append(
                {
                    "color": color,
                    "coords": global_coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "holes": _hole_count(mask),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each main-area object according to the hole-count→geometry mapping encoded in the right-side legend.

    Input structure:
    - Background is black.
    - A grey-lined vertical legend panel on the right is split into rows and two columns of cells; each row shows one prototype shape and its transformed counterpart.
    - The left side contains several separated colored 4-connected objects, each with 0 to 3 enclosed black holes.

    Rule:
    - Read the legend row by row: count the enclosed holes in the left prototype, compare the cropped left and right shapes, and infer which geometric transform corresponds to that hole count.
    - For every colored object in the main area, count its enclosed holes inside its own bounding box.
    - Apply the mapped transform to that object's pixel mask, keeping the transform centered on the object's bounding-box center and writing the transformed shape back in place.
    - Leave the legend panel unchanged.

    Key insight:
    - The legend does not label transforms directly; the solver must infer a topology-based lookup table from hole count to geometric operation, then reuse that table on different-colored objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, legend_bbox, bg, _line_color = _legend_mapping(g)
    legend_c0 = legend_bbox[1]

    objects = _extract_objects(g, legend_c0, bg)

    out = g.copy()
    for obj in objects:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = bg

    for obj in objects:
        assert obj["holes"] in mapping, f"missing legend entry for hole count {obj['holes']}"
        transformed = _apply_op(obj["mask"], mapping[obj["holes"]])
        r0, c0, r1, c1 = obj["bbox"]
        hh, ww = transformed.shape
        nr0 = (r0 + r1 - hh + 1) // 2
        nc0 = (c0 + c1 - ww + 1) // 2
        for rr, cc in np.argwhere(transformed):
            gr = nr0 + int(rr)
            gc = nc0 + int(cc)
            if 0 <= gr < out.shape[0] and 0 <= gc < legend_c0:
                out[gr, gc] = obj["color"]

    return out


def _mask_signature(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return mask.astype(np.uint8).tobytes(), tuple(mask.shape)


def _is_unique_under_ops(mask: np.ndarray) -> bool:
    sigs = {_mask_signature(_apply_op(mask, name)) for name in _OP_NAMES}
    return len(sigs) == len(_OP_NAMES)


def _adjacent_zeros(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dx, c + dy
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    random.shuffle(cells)
    return cells


def _mutate_template(base: np.ndarray, holes: int) -> np.ndarray:
    mask = base.copy()
    for _ in range(random.randint(0, 2)):
        changed = False
        for r, c in _adjacent_zeros(mask):
            trial = mask.copy()
            trial[r, c] = True
            if _hole_count(trial) == holes and _is_unique_under_ops(trial):
                mask = trial
                changed = True
                break
        if not changed:
            break
    return mask


def _place_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color
    grid[top : top + h, left : left + w] = sub


def _legend_separators() -> tuple[list[int], list[int]]:
    rows = [0]
    for _ in range(_ROW_COUNT):
        rows.append(rows[-1] + _CELL + 1)
    cols = [0, _CELL + 1, 2 * _CELL + 2]
    return rows, cols


def _draw_legend(
    grid: np.ndarray,
    main_width: int,
    mapping: dict[int, str],
) -> None:
    rows, cols = _legend_separators()
    panel = grid[:, main_width:]

    panel[rows, :] = _GREY
    panel[:, cols] = _GREY

    hole_order = random.sample([0, 1, 2, 3], 4)
    for row_idx, holes in enumerate(hole_order):
        left_mask = _mutate_template(_BASE_MASKS[holes], holes)
        left_mask = _apply_op(left_mask, random.choice(_OP_NAMES))
        right_mask = _apply_op(left_mask, mapping[holes])

        rs = rows[row_idx] + 1
        left_cs = cols[0] + 1
        right_cs = cols[1] + 1
        _place_mask(panel, rs, left_cs, left_mask, _LEGEND_FG)
        _place_mask(panel, rs, right_cs, right_mask, _LEGEND_FG)


def _sample_slots(main_width: int) -> list[tuple[int, int]]:
    row_offset = random.randint(0, 2)
    rows = [row_offset + 6 * i for i in range(4)]
    col_offset = random.randint(0, main_width - 11)
    cols = [col_offset, col_offset + 6]
    return [(r, c) for r in rows for c in cols]


def _gen_attempt() -> np.ndarray | None:
    main_width = random.randint(13, 17)
    grid = np.full((_LEGEND_H, main_width + _LEGEND_W), _BG, dtype=int)

    chosen_ops = random.sample(_OP_NAMES, 4)
    random.shuffle(chosen_ops)
    mapping = {holes: chosen_ops[holes] for holes in range(4)}
    _draw_legend(grid, main_width, mapping)

    slots = _sample_slots(main_width)
    obj_count = random.randint(4, 7)
    random.shuffle(slots)
    slot_choices = slots[:obj_count]

    hole_values = [0, 1, 2, 3]
    while len(hole_values) < obj_count:
        hole_values.append(random.randint(0, 3))
    random.shuffle(hole_values)

    colors = random.sample(_OBJECT_COLORS, obj_count)
    for (top, left), holes, color in zip(slot_choices, hole_values, colors):
        mask = _mutate_template(_BASE_MASKS[holes], holes)
        mask = _apply_op(mask, random.choice(_OP_NAMES))
        _place_mask(grid, top, left, mask, color)

    parsed_mapping, legend_bbox, bg, line_color = _legend_mapping(grid)
    if parsed_mapping != mapping:
        return None
    if bg != _BG or line_color != _GREY or legend_bbox[1] != main_width:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if not np.array_equal(out[:, main_width:], grid[:, main_width:]):
        return None
    if np.any(out[:, main_width:] == bg) != np.any(grid[:, main_width:] == bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
