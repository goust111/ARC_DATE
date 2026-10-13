from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, crop_bbox, find_components, paste, random_free_location  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        assert monochromatic and return_format == "dict"
        if bg is None:
            bg = Color.BLACK
        color_list = colors if colors is not None else sorted(set(map(int, grid.flatten())) - {int(bg)})
        out = []
        for color in color_list:
            mask = grid == int(color)
            seen = np.zeros(mask.shape, dtype=bool)
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if not mask[r, c] or seen[r, c]:
                        continue
                    q = deque([(r, c)])
                    seen[r, c] = True
                    coords = []
                    while q:
                        rr, cc = q.popleft()
                        coords.append((rr, cc))
                        for nr, nc in _neighbors(rr, cc, connectivity):
                            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    coord_arr = np.array(coords, dtype=int)
                    out.append(
                        {
                            "color": int(color),
                            "coords": coord_arr,
                            "bbox": _bbox_coords(coord_arr),
                            "area": int(coord_arr.shape[0]),
                        }
                    )
        return out

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        r0 = int(r0)
        c0 = int(c0)
        h, w = patch.shape
        view = dst[r0 : r0 + h, c0 : c0 + w]
        use = np.ones(patch.shape, dtype=bool) if mask is None else mask.astype(bool)
        if transparent is not None:
            use &= patch != int(transparent)
        view[use] = patch[use]
        return dst

    def random_free_location(
        canvas: np.ndarray,
        patch: np.ndarray,
        *,
        transparent: int,
        border: int = 0,
        padding: int = 0,
        padding_connectivity: int = 8,
    ) -> tuple[int, int]:
        occ = canvas != int(transparent)
        shape_mask = patch != int(transparent)
        h, w = patch.shape
        positions = []
        for r0 in range(int(border), canvas.shape[0] - h - int(border) + 1):
            for c0 in range(int(border), canvas.shape[1] - w - int(border) + 1):
                block = occ[r0 : r0 + h, c0 : c0 + w]
                if np.any(block & shape_mask):
                    continue
                if padding:
                    rr0 = max(0, r0 - padding)
                    cc0 = max(0, c0 - padding)
                    rr1 = min(canvas.shape[0], r0 + h + padding)
                    cc1 = min(canvas.shape[1], c0 + w + padding)
                    if np.any(occ[rr0:rr1, cc0:cc1]):
                        continue
                positions.append((r0, c0))
        if not positions:
            raise ValueError("no free location")
        return random.choice(positions)


BG = Color.BLACK
FRAME = Color.GREY
PALETTE = [c for c in getattr(Color, "NOT_BLACK", [1, 2, 3, 4, 5, 6, 7, 8, 9]) if c != FRAME]

FRAME_COUNT_RANGE = (3, 5)
CAPACITY_VALUES = list(range(3, 10))
INVALID_COUNT_VALUES = list(range(4, 11))
CANVAS_RANGE = (18, 30)
PLACE_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200
GENERATE_ATTEMPTS = 300


def _neighbors(r: int, c: int, connectivity: int = 4):
    if int(connectivity) == 8:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                yield int(r) + dr, int(c) + dc
    else:
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


def _boundary_reachable(open_mask: np.ndarray) -> np.ndarray:
    open_mask = np.asarray(open_mask, dtype=bool)
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, 4):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _component_count(mask: np.ndarray, connectivity: int = 4) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            total += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return int(total)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _enclosed_mask(frame_mask: np.ndarray) -> np.ndarray:
    open_mask = ~np.asarray(frame_mask, dtype=bool)
    return open_mask & ~_boundary_reachable(open_mask)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, colors=[FRAME], connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one gray frame"

    infos = []
    covered_non_frame = np.zeros(g.shape, dtype=bool)
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(g, bbox)
        frame_mask = sub == FRAME
        interior_mask = _enclosed_mask(frame_mask)
        assert np.any(interior_mask), "every frame must enclose interior cells"

        colors = sorted(set(map(int, sub[interior_mask].reshape(-1))) - {BG})
        assert len(colors) == 1, "each frame must contain exactly one instruction color"
        color = int(colors[0])
        count = int(np.sum(sub[interior_mask] == color))
        capacity = int(np.sum(sub[interior_mask] == BG))
        assert count > 0 and capacity > 0

        r0, c0, r1, c1 = bbox
        covered_non_frame[r0 : r1 + 1, c0 : c1 + 1] |= interior_mask
        infos.append(
            {
                "bbox": bbox,
                "frame_mask": frame_mask,
                "interior_mask": interior_mask,
                "capacity": capacity,
                "count": count,
                "color": color,
            }
        )

    stray = (g != BG) & (g != FRAME) & ~covered_non_frame
    assert not np.any(stray), "instruction pixels must be inside frames"
    infos.sort(key=lambda info: info["bbox"])
    capacities = [info["capacity"] for info in infos]
    assert len(set(capacities)) == len(capacities), "frame capacities must be unique"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's dot-count as an address and fill the uniquely matching-capacity frame with that color.

    Input structure:
    - The background is black and several disconnected gray hollow frames appear on it.
    - Each frame encloses one instruction group: a set of same-color pixels scattered inside the frame.
    - Different frames have different interior black-cell capacities, and exactly one instruction count matches none of them.

    Rule:
    - For every gray frame, identify its enclosed interior, count its black cells as the frame capacity, and count the instruction pixels of the single non-gray color inside it.
    - If an instruction count equals some frame's capacity, fill that target frame's entire interior with the instruction color, overwriting any pixels already inside that frame.
    - If an instruction count matches no frame capacity, leave that instruction and its frame unchanged; keep all gray frame walls unchanged.

    Key insight:
    - The dots do not describe their own container by width, height, or perimeter; they are indirect pointers whose only valid key is equality between dot-count and another frame's interior black-cell capacity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    infos = _extract_frames(g)
    capacity_to_index = {info["capacity"]: idx for idx, info in enumerate(infos)}

    invalid_sources = []
    hit_counts = [0] * len(infos)
    assignments: list[tuple[int, int]] = []
    for source_idx, info in enumerate(infos):
        target_idx = capacity_to_index.get(info["count"])
        if target_idx is None:
            invalid_sources.append(source_idx)
            continue
        hit_counts[target_idx] += 1
        assignments.append((source_idx, target_idx))

    assert len(invalid_sources) == 1, "expected exactly one invalid instruction"
    assert sum(hit_counts) == len(infos) - 1, "all non-invalid instructions must be valid"
    assert max(hit_counts) == 1, "each target frame must be addressed at most once"
    assert hit_counts[invalid_sources[0]] == 0, "the invalid source frame must be the unique unaddressed frame"

    out = g.copy()
    for source_idx, target_idx in assignments:
        target = infos[target_idx]
        r0, c0, r1, c1 = target["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][target["interior_mask"]] = int(infos[source_idx]["color"])
    return out


def _mask_from_spec(spec: dict) -> np.ndarray:
    h = int(spec["h"])
    w = int(spec["w"])
    mask = np.ones((h, w), dtype=bool)
    if spec["kind"] == "rect":
        return mask
    a = int(spec["a"])
    b = int(spec["b"])
    if spec["kind"] == "corner":
        corner = spec["corner"]
        if corner == "tl":
            mask[:a, :b] = False
        elif corner == "tr":
            mask[:a, w - b :] = False
        elif corner == "bl":
            mask[h - a :, :b] = False
        else:
            mask[h - a :, w - b :] = False
    else:
        edge = spec["edge"]
        offset = int(spec["offset"])
        if edge == "top":
            mask[:a, offset : offset + b] = False
        elif edge == "bottom":
            mask[h - a :, offset : offset + b] = False
        elif edge == "left":
            mask[offset : offset + a, :b] = False
        else:
            mask[offset : offset + a, w - b :] = False
    return np.rot90(mask, int(spec["rot"]))


def _shape_specs(
    area: int,
    *,
    require_irregular: bool,
    preferred_inner_dim: int | None = None,
) -> list[dict]:
    specs = []
    for h in range(2, 7):
        for w in range(2, 7):
            if preferred_inner_dim is not None and preferred_inner_dim not in (h, w):
                continue
            deficit = h * w - int(area)
            if deficit < 0:
                continue
            if deficit == 0:
                if require_irregular:
                    continue
                for rot in range(4):
                    specs.append({"kind": "rect", "h": h, "w": w, "rot": rot})
                continue

            for a in range(1, h):
                if deficit % a != 0:
                    continue
                b = deficit // a
                if not (1 <= b < w):
                    continue

                for corner in ("tl", "tr", "bl", "br"):
                    for rot in range(4):
                        specs.append(
                            {
                                "kind": "corner",
                                "h": h,
                                "w": w,
                                "a": a,
                                "b": b,
                                "corner": corner,
                                "rot": rot,
                            }
                        )

                if w - b >= 2:
                    for offset in range(1, w - b):
                        for edge in ("top", "bottom"):
                            for rot in range(4):
                                specs.append(
                                    {
                                        "kind": "edge",
                                        "h": h,
                                        "w": w,
                                        "a": a,
                                        "b": b,
                                        "edge": edge,
                                        "offset": offset,
                                        "rot": rot,
                                    }
                                )
                if h - a >= 2:
                    for offset in range(1, h - a):
                        for edge in ("left", "right"):
                            for rot in range(4):
                                specs.append(
                                    {
                                        "kind": "edge",
                                        "h": h,
                                        "w": w,
                                        "a": a,
                                        "b": b,
                                        "edge": edge,
                                        "offset": offset,
                                        "rot": rot,
                                    }
                                )
    random.shuffle(specs)
    return specs


def _build_frame_patch(
    area: int,
    count: int,
    color: int,
    *,
    require_irregular: bool,
    preferred_inner_dim: int | None = None,
) -> dict | None:
    for spec in _shape_specs(area, require_irregular=require_irregular, preferred_inner_dim=preferred_inner_dim):
        inner = _mask_from_spec(spec)
        assert int(inner.sum()) == int(area)

        padded_inner = np.pad(inner, 1, constant_values=False)
        frame_mask = _dilate8(padded_inner) & ~padded_inner
        if _component_count(frame_mask, 4) != 1:
            continue
        if not np.array_equal(_enclosed_mask(frame_mask), padded_inner):
            continue

        coords = np.argwhere(padded_inner)
        if len(coords) <= count:
            continue
        chosen = coords[np.random.choice(len(coords), size=count, replace=False)]

        patch = np.full(padded_inner.shape, BG, dtype=int)
        patch[frame_mask] = FRAME
        patch[chosen[:, 0], chosen[:, 1]] = int(color)
        return {
            "patch": patch,
            "inner_mask": padded_inner,
            "frame_mask": frame_mask,
            "count": int(count),
            "capacity": int(area - count),
            "color": int(color),
            "irregular": bool(int(inner.sum()) != inner.shape[0] * inner.shape[1]),
        }
    return None


def _choose_permutation(valid_sources: list[int], force_self: bool) -> dict[int, int] | None:
    targets = valid_sources[:]
    for _ in range(100):
        random.shuffle(targets)
        mapping = dict(zip(valid_sources, targets))
        has_self = any(src == dst for src, dst in mapping.items())
        if force_self and not has_self:
            continue
        if not force_self and has_self:
            continue
        return mapping
    return None


def _pick_invalid_count(capacities: list[int], force_mislead: bool) -> int | None:
    pool = [v for v in INVALID_COUNT_VALUES if v not in capacities]
    if not pool:
        return None
    if force_mislead:
        pool = [v for v in pool if 4 <= v <= 8]
        if not pool:
            return None
    return int(random.choice(pool))


def _render_expected(grid: np.ndarray, specs: list[dict], targets: list[int | None]) -> np.ndarray:
    out = grid.copy()
    for src_idx, dst_idx in enumerate(targets):
        if dst_idx is None:
            continue
        spec = specs[int(dst_idx)]
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h, w = spec["patch"].shape
        out[r0 : r0 + h, c0 : c0 + w][spec["inner_mask"]] = int(specs[src_idx]["color"])
    return out


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    capacities: list[int],
    counts: list[int],
    targets: list[int | None],
    invalid_idx: int,
    *,
    force_self: bool,
    force_irregular: bool,
    force_mislead: bool,
    invalid_count: int,
) -> bool:
    try:
        infos = _extract_frames(grid)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if len(infos) != len(capacities):
        return False

    got_caps = sorted(info["capacity"] for info in infos)
    if got_caps != sorted(capacities):
        return False
    if sorted(info["count"] for info in infos) != sorted(counts):
        return False

    capacity_to_index = {info["capacity"]: idx for idx, info in enumerate(infos)}
    observed_targets = []
    invalids = []
    for idx, info in enumerate(infos):
        dst = capacity_to_index.get(info["count"])
        observed_targets.append(dst)
        if dst is None:
            invalids.append(idx)
    if len(invalids) != 1:
        return False
    invalid_info_idx = int(invalids[0])
    if infos[invalid_info_idx]["count"] != invalid_count:
        return False
    if sum(dst is not None for dst in observed_targets) != len(infos) - 1:
        return False
    if sum(dst == invalid_info_idx for dst in observed_targets if dst is not None) != 0:
        return False
    if len({dst for dst in observed_targets if dst is not None}) != len(infos) - 1:
        return False

    has_self = any(dst == idx for idx, dst in enumerate(observed_targets) if dst is not None)
    if force_self != has_self:
        return False
    if force_irregular and not any(int(info["interior_mask"].sum()) != info["interior_mask"].shape[0] * info["interior_mask"].shape[1] for info in infos):
        return False
    if force_mislead:
        attrs = set()
        for info in infos:
            r0, c0, r1, c1 = info["bbox"]
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            attrs.update({h, w, 2 * (h + w) - 4})
        if invalid_count not in attrs:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    capacities = random.sample(CAPACITY_VALUES, n_frames)
    invalid_idx = random.randrange(n_frames)

    force_self = random.random() < 0.6
    force_irregular = random.random() < 0.75
    force_mislead = random.random() < 0.55

    invalid_count = _pick_invalid_count(capacities, force_mislead)
    if invalid_count is None:
        return None

    valid_sources = [idx for idx in range(n_frames) if idx != invalid_idx]
    mapping = _choose_permutation(valid_sources, force_self=force_self)
    if mapping is None:
        return None

    targets: list[int | None] = [None] * n_frames
    counts = [0] * n_frames
    for src_idx, dst_idx in mapping.items():
        targets[src_idx] = int(dst_idx)
        counts[src_idx] = int(capacities[dst_idx])
    counts[invalid_idx] = int(invalid_count)

    colors = random.sample(PALETTE, n_frames)
    mislead_shape_idx = random.randrange(n_frames) if force_mislead else None
    preferred_inner_dim = invalid_count - 2 if force_mislead else None

    specs = []
    irregular_assigned = False
    for idx in range(n_frames):
        area = int(capacities[idx] + counts[idx])
        require_irregular = force_irregular and (not irregular_assigned or idx == n_frames - 1)
        dim = preferred_inner_dim if idx == mislead_shape_idx else None
        spec = _build_frame_patch(
            area,
            counts[idx],
            colors[idx],
            require_irregular=require_irregular,
            preferred_inner_dim=dim,
        )
        if spec is None:
            return None
        irregular_assigned = irregular_assigned or spec["irregular"]
        specs.append(spec)
    if force_irregular and not irregular_assigned:
        return None

    total_area = sum(int(np.count_nonzero(spec["patch"] != BG)) for spec in specs)
    max_h = max(spec["patch"].shape[0] for spec in specs)
    max_w = max(spec["patch"].shape[1] for spec in specs)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 3))) + 4)
    if min_side > CANVAS_RANGE[1]:
        return None

    height = random.randint(max(CANVAS_RANGE[0], min_side), CANVAS_RANGE[1])
    width = random.randint(max(CANVAS_RANGE[0], min_side), CANVAS_RANGE[1])
    grid = np.full((height, width), BG, dtype=int)

    order = list(range(n_frames))
    random.shuffle(order)
    order.sort(key=lambda idx: specs[idx]["patch"].shape[0] * specs[idx]["patch"].shape[1], reverse=True)
    for idx in order:
        spec = specs[idx]
        try:
            r0, c0 = random_free_location(
                grid,
                spec["patch"],
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, spec["patch"], r0, c0, transparent=BG)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)

    expected = _render_expected(grid, specs, targets)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_ok(
        grid,
        expected,
        capacities,
        counts,
        targets,
        invalid_idx,
        force_self=force_self,
        force_irregular=force_irregular,
        force_mislead=force_mislead,
        invalid_count=invalid_count,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input: failed to build a valid capacity-addressing instance")
