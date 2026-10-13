from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
CANDIDATE = 1
MATCHED = 3
OBSTACLE = 5
PROTOTYPE = 7
MAX_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    trimmed = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, trimmed.shape[1]):
        if not np.array_equal(trimmed[:, c], trimmed[:, c - 1]):
            col_keep.append(c)
    return trimmed[:, col_keep]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _mask_key(_compress_runs(mask))


def _orientation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.copy())
    return variants


def _signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_signature(variant) for variant in _orientation_variants(mask)}


def _signature_class(mask: np.ndarray) -> frozenset[tuple[tuple[int, ...], ...]]:
    return frozenset(_signature_set(mask))


def _raw_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    return any(a.shape == variant.shape and np.array_equal(a, variant) for variant in _orientation_variants(b))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    target = {tuple(int(v) for v in rc) for rc in coords}
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(target)


def _stretch_variants(mask: np.ndarray, *, max_area: int, max_dim: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for oriented in _orientation_variants(mask):
        row_choices = list(itertools.product((1, 2), repeat=oriented.shape[0]))
        col_choices = list(itertools.product((1, 2), repeat=oriented.shape[1]))
        for row_factors in row_choices:
            for col_factors in col_choices:
                stretched = np.repeat(np.repeat(oriented, row_factors, axis=0), col_factors, axis=1)
                if max(stretched.shape) > max_dim:
                    continue
                area = int(stretched.sum())
                if not 3 <= area <= max_area:
                    continue
                key = _mask_key(stretched)
                if key in seen:
                    continue
                seen.add(key)
                out.append(stretched.copy())
    return out


def _build_core_library() -> list[dict[str, object]]:
    classes: dict[frozenset[tuple[tuple[int, ...], ...]], np.ndarray] = {}
    for h in range(2, 5):
        for w in range(2, 5):
            cells = list(range(h * w))
            for area in range(3, min(6, h * w) + 1):
                for choice in itertools.combinations(cells, area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in choice:
                        mask[idx // w, idx % w] = True
                    if bool(mask.all()):
                        continue
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if not _is_connected(mask):
                        continue
                    compressed = _compress_runs(mask)
                    if min(compressed.shape) < 2:
                        continue
                    sig_class = _signature_class(mask)
                    classes.setdefault(sig_class, mask.copy())

    library: list[dict[str, object]] = []
    for sig_class, core in classes.items():
        candidate_variants = _stretch_variants(core, max_area=8, max_dim=4)
        prototype_variants = [variant for variant in candidate_variants if int(variant.sum()) <= 6]
        if prototype_variants and candidate_variants:
            library.append(
                {
                    "core": core,
                    "sig_class": sig_class,
                    "prototype_variants": prototype_variants,
                    "candidate_variants": candidate_variants,
                }
            )
    return library


CORE_LIBRARY = _build_core_library()
ALL_VARIANTS = [
    {
        "sig_class": entry["sig_class"],
        "mask": variant,
        "area": int(variant.sum()),
        "shape": variant.shape,
    }
    for entry in CORE_LIBRARY
    for variant in entry["candidate_variants"]
]


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[tuple[int, ...], ...]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    proto_comps = _components_mask(g == PROTOTYPE)
    cand_comps = _components_mask(g == CANDIDATE)
    assert len(proto_comps) == 1
    assert cand_comps
    proto_coords = proto_comps[0]
    proto_mask = _component_mask(proto_coords)
    proto_sigs = _signature_set(proto_mask)
    candidates: list[dict[str, object]] = []
    for coords in cand_comps:
        mask = _component_mask(coords)
        candidates.append(
            {
                "coords": coords,
                "mask": mask,
                "is_match": _signature(mask) in proto_sigs,
                "key": _coords_key(coords),
            }
        )
    return proto_coords, proto_sigs, candidates


def _fits(coords: np.ndarray, occ: np.ndarray) -> bool:
    h, w = occ.shape
    if int(coords[:, 0].min()) < 0 or int(coords[:, 1].min()) < 0:
        return False
    if int(coords[:, 0].max()) >= h or int(coords[:, 1].max()) >= w:
        return False
    return not bool(occ[coords[:, 0], coords[:, 1]].any())


def _buffer_clear(coords: np.ndarray, occ: np.ndarray) -> bool:
    below = coords + np.array([1, 0], dtype=int)
    if int(below[:, 0].max()) >= occ.shape[0]:
        return False
    return not bool(occ[below[:, 0], below[:, 1]].any())


def _simulate(grid: np.ndarray, *, stack_obstacles: bool = True) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    _proto_coords, _proto_sigs, candidates = _parse_scene(g)
    out = g.copy()
    out[out == CANDIDATE] = BG
    occ = out != BG

    order = sorted(
        range(len(candidates)),
        key=lambda idx: (
            int(candidates[idx]["coords"][:, 0].max()),
            int(candidates[idx]["coords"][:, 0].min()),
            int(candidates[idx]["coords"][:, 1].min()),
        ),
        reverse=True,
    )

    infos: list[dict[str, object] | None] = [None] * len(candidates)
    for idx in order:
        obj = candidates[idx]
        final_coords = np.array(obj["coords"], copy=True)
        while True:
            trial = final_coords + np.array([1, 0], dtype=int)
            if not _fits(trial, occ):
                break
            if bool(obj["is_match"]) and not _buffer_clear(trial, occ):
                break
            final_coords = trial

        color = MATCHED if bool(obj["is_match"]) else CANDIDATE
        out[final_coords[:, 0], final_coords[:, 1]] = color
        if stack_obstacles:
            occ[final_coords[:, 0], final_coords[:, 1]] = True
        infos[idx] = {
            "orig_key": obj["key"],
            "is_match": bool(obj["is_match"]),
            "final_coords": final_coords,
        }

    return out, [info for info in infos if info is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use run-compressed shape matching to decide whether each falling candidate lands buffered or flush.

    Input structure:
    - Black background with a full-width gray obstacle occupying the bottom one or two rows.
    - A single orange prototype object stays near the upper-left.
    - Several blue candidate objects sit above the obstacle and may share column ranges.

    Rule:
    - Crop the prototype and each candidate to its bbox, then collapse adjacent identical rows and columns.
    - A candidate whose compressed mask matches the prototype up to rotation or mirror is a match.
    - Move candidates downward in order from lower original position to higher one.
    - Matches stop with one blank buffer row above the first obstacle they would otherwise touch; non-matches stop flush.
    - Already settled candidates, the fixed prototype, and the gray barrier all act as obstacles.
    - Recolor matched settled candidates green and leave non-matches blue.

    Key insight:
    - The landing-height difference is controlled by run-compressed shape equivalence, not by raw size or raw orientation.
    """
    out, _infos = _simulate(np.asarray(grid, dtype=int), stack_obstacles=True)
    return out


def _pick_prototype_and_special_match(rng: random.Random) -> tuple[dict[str, object], np.ndarray, np.ndarray] | None:
    entries = CORE_LIBRARY[:]
    rng.shuffle(entries)
    for entry in entries:
        proto_variants = list(entry["prototype_variants"])
        cand_variants = list(entry["candidate_variants"])
        rng.shuffle(proto_variants)
        rng.shuffle(cand_variants)
        for proto_mask in proto_variants:
            for match_mask in cand_variants:
                if not _raw_equivalent(match_mask, proto_mask):
                    return entry, proto_mask.copy(), match_mask.copy()
    return None


def _pick_match_mask(
    rng: random.Random,
    entry: dict[str, object],
    proto_mask: np.ndarray,
    *,
    require_non_raw: bool = False,
) -> np.ndarray | None:
    options = []
    for variant in entry["candidate_variants"]:
        if require_non_raw and _raw_equivalent(variant, proto_mask):
            continue
        options.append(variant)
    if not options:
        return None
    return np.array(rng.choice(options), copy=True)


def _pick_distractor_mask(rng: random.Random, proto_sig_class: object, target_mask: np.ndarray) -> np.ndarray | None:
    target_area = int(target_mask.sum())
    target_h, target_w = target_mask.shape
    pool = [
        item
        for item in ALL_VARIANTS
        if item["sig_class"] != proto_sig_class
    ]
    if not pool:
        return None
    rng.shuffle(pool)
    pool.sort(
        key=lambda item: (
            abs(int(item["area"]) - target_area),
            abs(int(item["shape"][0]) - target_h),
            abs(int(item["shape"][1]) - target_w),
        )
    )
    top = pool[: min(30, len(pool))]
    return np.array(rng.choice(top)["mask"], copy=True)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + int(top)
    cols = coords[:, 1] + int(left)
    grid[rows, cols] = int(color)
    return np.column_stack([rows, cols]).astype(int)


def _barrier_top(grid: np.ndarray) -> int:
    for r in range(grid.shape[0]):
        if bool(np.all(grid[r] == OBSTACLE)):
            return int(r)
    raise AssertionError("missing obstacle row")


def _candidate_roles(n: int, rng: random.Random) -> list[list[dict[str, object]]]:
    remaining = n - 4
    lanes: list[list[dict[str, object]]] = [
        [{"role": "standalone_match"}],
        [{"role": "standalone_unmatch"}],
        [{"role": "stack_a_lower"}, {"role": "stack_a_upper"}],
    ]
    if remaining == 1:
        lanes.append([{"role": "extra_0"}])
    elif remaining == 2:
        if rng.random() < 0.5:
            lanes.append([{"role": "stack_b_lower"}, {"role": "stack_b_upper"}])
        else:
            lanes.append([{"role": "extra_0"}])
            lanes.append([{"role": "extra_1"}])
    return lanes


def _assign_masks(rng: random.Random, n: int) -> tuple[np.ndarray, dict[str, object], list[list[dict[str, object]]]] | None:
    picked = _pick_prototype_and_special_match(rng)
    if picked is None:
        return None
    entry, proto_mask, special_match = picked
    proto_sig_class = entry["sig_class"]

    lanes = _candidate_roles(n, rng)
    role_masks: dict[str, dict[str, object]] = {
        "standalone_match": {"mask": special_match, "is_match": True, "require_non_raw": True},
    }
    distractor = _pick_distractor_mask(rng, proto_sig_class, special_match)
    if distractor is None:
        return None
    role_masks["standalone_unmatch"] = {"mask": distractor, "is_match": False}

    for lane in lanes:
        for item in lane:
            role = str(item["role"])
            if role in role_masks:
                continue
            want_match = rng.random() < 0.5
            if want_match:
                mask = _pick_match_mask(rng, entry, proto_mask, require_non_raw=False)
                if mask is None:
                    return None
                role_masks[role] = {"mask": mask, "is_match": True}
            else:
                target = role_masks["standalone_unmatch"]["mask"]
                mask = _pick_distractor_mask(rng, proto_sig_class, target)
                if mask is None:
                    return None
                role_masks[role] = {"mask": mask, "is_match": False}

    for lane in lanes:
        for item in lane:
            item.update(role_masks[str(item["role"])])

    return proto_mask, entry, lanes


def _layout_scene(
    rng: random.Random,
    proto_mask: np.ndarray,
    lanes: list[list[dict[str, object]]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    barrier_h = rng.choice([1, 2])
    rows = rng.randint(18, 24)
    barrier_top = rows - barrier_h
    proto_top = rng.randint(1, 2)
    proto_left = rng.randint(1, 2)

    lane_widths = [max(int(item["mask"].shape[1]) for item in lane) for lane in lanes]
    left_start = proto_left + proto_mask.shape[1] + rng.randint(2, 4)
    gaps = [rng.randint(2, 3) for _ in range(max(0, len(lanes) - 1))]
    min_cols = left_start + sum(lane_widths) + sum(gaps) + 1
    if min_cols > 30:
        return None
    cols = rng.randint(max(14, min_cols), min(30, min_cols + 4))
    extra = cols - min_cols
    slacks = [0] * (len(lanes) + 1)
    for _ in range(extra):
        slacks[rng.randrange(len(slacks))] += 1

    order = list(range(len(lanes)))
    rng.shuffle(order)
    lanes = [lanes[i] for i in order]
    lane_widths = [lane_widths[i] for i in order]
    gaps = [rng.randint(2, 3) for _ in range(max(0, len(lanes) - 1))]

    lane_lefts: list[int] = []
    x = left_start + slacks[0]
    for i, width in enumerate(lane_widths):
        lane_lefts.append(int(x))
        x += width
        if i < len(gaps):
            x += gaps[i] + slacks[i + 1]
    if x > cols - 1:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    grid[barrier_top:, :] = OBSTACLE
    _paint_mask(grid, proto_mask, proto_top, proto_left, PROTOTYPE)

    placed: list[dict[str, object]] = []
    upper_limit = rows // 2
    for lane, lane_left, lane_width in zip(lanes, lane_lefts, lane_widths):
        if len(lane) == 1:
            item = lane[0]
            mask = np.array(item["mask"], copy=True)
            h, w = mask.shape
            top_max = upper_limit - h - 1
            if top_max < 1:
                return None
            top = rng.randint(1, top_max)
            left = lane_left + rng.randint(0, lane_width - w)
            coords = _paint_mask(grid, mask, top, left, CANDIDATE)
            placed.append({"role": item["role"], "coords": coords, "mask": mask, "is_match": bool(item["is_match"])})
            continue

        lower = lane[0]
        upper = lane[1]
        lower_mask = np.array(lower["mask"], copy=True)
        upper_mask = np.array(upper["mask"], copy=True)
        lower_h, lower_w = lower_mask.shape
        upper_h, upper_w = upper_mask.shape
        gap = rng.randint(2, 4)
        lower_top_min = max(upper_h + gap + 1, 4)
        lower_top_max = upper_limit - lower_h - 1
        if lower_top_max < lower_top_min:
            return None
        lower_top = rng.randint(lower_top_min, lower_top_max)
        upper_top = lower_top - gap - upper_h
        if upper_top < 1:
            return None
        lower_left = lane_left + (lane_width - lower_w) // 2
        upper_left = lane_left + (lane_width - upper_w) // 2
        lower_coords = _paint_mask(grid, lower_mask, lower_top, lower_left, CANDIDATE)
        upper_coords = _paint_mask(grid, upper_mask, upper_top, upper_left, CANDIDATE)
        placed.append({"role": lower["role"], "coords": lower_coords, "mask": lower_mask, "is_match": bool(lower["is_match"])})
        placed.append({"role": upper["role"], "coords": upper_coords, "mask": upper_mask, "is_match": bool(upper["is_match"])})

    return grid, placed


def _validate_scene(grid: np.ndarray, proto_mask: np.ndarray, placed: list[dict[str, object]]) -> bool:
    solved = main(grid)
    expected, actual_infos = _simulate(grid, stack_obstacles=True)
    naive, naive_infos = _simulate(grid, stack_obstacles=False)
    if not np.array_equal(solved, expected):
        return False

    barrier_top = _barrier_top(grid)
    proto_sigs = _signature_set(proto_mask)
    role_by_key = { _coords_key(item["coords"]): str(item["role"]) for item in placed }
    mask_by_role = { str(item["role"]): np.array(item["mask"], copy=True) for item in placed }
    info_by_role = { role_by_key[info["orig_key"]]: info for info in actual_infos }
    naive_by_role = { role_by_key[info["orig_key"]]: info for info in naive_infos }

    if "standalone_match" not in info_by_role or "standalone_unmatch" not in info_by_role:
        return False
    match_bottom = int(info_by_role["standalone_match"]["final_coords"][:, 0].max())
    unmatch_bottom = int(info_by_role["standalone_unmatch"]["final_coords"][:, 0].max())
    if match_bottom != barrier_top - 2:
        return False
    if unmatch_bottom != barrier_top - 1:
        return False

    special_match_mask = mask_by_role["standalone_match"]
    if _raw_equivalent(special_match_mask, proto_mask):
        return False
    if _signature(special_match_mask) not in proto_sigs:
        return False
    if _signature(mask_by_role["standalone_unmatch"]) in proto_sigs:
        return False

    stack_roles = [role for role in info_by_role if role.startswith("stack_")]
    if not stack_roles:
        return False
    if not any(
        not np.array_equal(info_by_role[role]["final_coords"], naive_by_role[role]["final_coords"])
        for role in stack_roles
    ):
        return False

    if not bool((solved == MATCHED).any()) or not bool((solved == CANDIDATE).any()):
        return False
    return not np.array_equal(expected, naive)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_candidates = rng.randint(4, 6)
    assigned = _assign_masks(rng, n_candidates)
    if assigned is None:
        return None
    proto_mask, _entry, lanes = assigned
    laid_out = _layout_scene(rng, proto_mask, lanes)
    if laid_out is None:
        return None
    grid, placed = laid_out

    if not (4 <= len(placed) <= 6):
        return None
    if sum(int(item["is_match"]) for item in placed) < 1:
        return None
    if sum(not bool(item["is_match"]) for item in placed) < 1:
        return None
    if not _validate_scene(grid, proto_mask, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-buffer ARC input")
