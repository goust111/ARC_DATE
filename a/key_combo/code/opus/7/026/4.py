from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
GREY = 5
CONTROL_COLORS = (1, 2, 3, 4, 7)
ALL_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cells_from_bbox_and_missing(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _missing_corner(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 == r0 + 1 and c1 == c0 + 1, "grey object must occupy a 2x2 bounding box"
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = set(ALL_CORNERS) - rel
    assert len(missing) == 1, "grey object must be an L triomino"
    return next(iter(missing))


def _direction_from_missing(missing: tuple[int, int]) -> tuple[int, int]:
    mr, mc = missing
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _move_cells(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _control_columns(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, _w = grid.shape
    assert h >= 3, "grid must have at least 3 rows"
    controls = {}
    for c, value in enumerate(grid[0].tolist()):
        value = int(value)
        if value == bg:
            continue
        assert value != GREY, "top-row controls cannot be grey"
        controls[c] = value
    assert 3 <= len(controls) <= 5, "expected 3 to 5 top-row control pixels"
    assert len(set(controls.values())) == len(controls), "control colors must be distinct"
    non_bg = np.argwhere(grid != bg)
    extra = [(int(r), int(c)) for r, c in non_bg if int(r) != 0 and int(grid[r, c]) != GREY]
    assert not extra, "only top-row controls and grey triominoes are allowed in the input"
    return controls


def _span_matches(cols: list[int], c0: int, c1: int) -> list[int]:
    return [c for c in cols if c0 <= c <= c1]


def _analyze_grid(grid: np.ndarray, *, require_generator_checks: bool = False) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    controls = _control_columns(g, bg)
    control_cols = sorted(controls)

    grey_components = _components_mask(g == GREY)
    assert 3 <= len(grey_components) <= 5, "expected 3 to 5 grey L triominoes"

    moved_records: list[dict] = []
    moved_seen: set[tuple[int, int]] = set()
    used_control_cols: set[int] = set()
    relation_kinds: set[str] = set()

    for comp in grey_components:
        assert comp.shape[0] == 3, "each grey object must have area 3"
        missing = _missing_corner(comp)
        dr, dc = _direction_from_missing(missing)
        source_box = _bbox(comp)
        moved_coords = _move_cells(tuple(map(tuple, comp.tolist())), dr, dc)

        for r, c in moved_coords:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "translated triomino must stay in bounds"
            assert (r, c) not in moved_seen, "translated triominoes must not overlap"
            assert not (r == 0 and c in controls), "translated triomino must not overlap a control pixel"
        moved_seen.update(moved_coords)

        moved_arr = np.array(moved_coords, dtype=int)
        _mr0, mc0, _mr1, mc1 = _bbox(moved_arr)
        moved_matches = _span_matches(control_cols, mc0, mc1)
        assert len(moved_matches) == 1, "translated column span must contain exactly one control column"
        target_col = moved_matches[0]
        used_control_cols.add(target_col)

        _sr0, sc0, _sr1, sc1 = source_box
        source_matches = _span_matches(control_cols, sc0, sc1)
        if len(source_matches) == 1 and source_matches[0] == target_col:
            relation = "same"
        elif len(source_matches) == 1:
            relation = "different"
        else:
            assert len(source_matches) == 0, "source span must not contain multiple control columns"
            relation = "none"
        relation_kinds.add(relation)

        moved_records.append(
            {
                "source_coords": tuple(sorted((int(r), int(c)) for r, c in comp.tolist())),
                "moved_coords": tuple(sorted((int(r), int(c)) for r, c in moved_coords)),
                "missing": missing,
                "target_col": target_col,
                "target_color": controls[target_col],
                "relation": relation,
            }
        )

    if require_generator_checks:
        assert used_control_cols == set(control_cols), "every control must color at least one translated triomino"
        assert "different" in relation_kinds, "at least one object must change control assignment after translation"
        assert relation_kinds & {"same", "none"}, "not every object may rely on the same changed-mapping case"

    return {
        "grid": g,
        "bg": bg,
        "controls": controls,
        "records": moved_records,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each grey L triomino one diagonal step toward its missing corner, then recolor it from the unique top-row control column inside its translated span.

    Input structure:
    - The background is black, and row 0 contains 3 to 5 isolated colored control pixels with distinct colors.
    - Below the top row, the only non-background objects are 3 to 5 grey 4-connected L triominoes, each occupying three cells of a 2x2 box.

    Rule:
    - For each grey triomino, identify which corner of its 2x2 bounding box is missing.
    - Translate the whole triomino by one diagonal step toward that missing corner.
    - Compute the translated triomino's bounding-box column span and find the unique control pixel whose column lies inside that span.
    - Erase the original grey triomino, repaint the translated triomino with that control pixel's color, and leave the controls unchanged.

    Key insight:
    - The missing corner encodes the motion direction, and that motion changes the column span that determines which control pixel supplies the final color.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    bg = info["bg"]
    controls = info["controls"]

    out = np.full_like(g, bg)
    for c, color in controls.items():
        out[0, c] = color
    for record in info["records"]:
        color = int(record["target_color"])
        for r, c in record["moved_coords"]:
            out[r, c] = color
    return out


def _expand8(cells: tuple[tuple[int, int], ...], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))
        blocked.add((r, c))
    return blocked


def _choose_control_columns(width: int, count: int) -> list[int] | None:
    slots = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(slots, count))
        gaps = [b - a for a, b in zip(cols, cols[1:], strict=False)]
        if gaps and min(gaps) >= 2 and any(g == 2 for g in gaps):
            return cols
    return None


def _relation_for_candidate(control_cols: list[int], source_c0: int, moved_c0: int, target_col: int) -> str | None:
    moved_matches = _span_matches(control_cols, moved_c0, moved_c0 + 1)
    if moved_matches != [target_col]:
        return None
    source_matches = _span_matches(control_cols, source_c0, source_c0 + 1)
    if len(source_matches) == 1 and source_matches[0] == target_col:
        return "same"
    if len(source_matches) == 1:
        return "different"
    if len(source_matches) == 0:
        return "none"
    return None


def _generate_candidates(height: int, width: int, control_cols: list[int]) -> list[dict]:
    candidates: list[dict] = []
    for missing in ALL_CORNERS:
        dr, dc = _direction_from_missing(missing)
        for target_col in control_cols:
            for side in (0, 1):
                moved_c0 = target_col - side
                source_c0 = moved_c0 - dc
                if moved_c0 < 0 or moved_c0 + 1 >= width:
                    continue
                if source_c0 < 0 or source_c0 + 1 >= width:
                    continue
                relation = _relation_for_candidate(control_cols, source_c0, moved_c0, target_col)
                if relation is None:
                    continue
                for source_r0 in range(2, height - 1):
                    moved_r0 = source_r0 + dr
                    if moved_r0 < 1 or moved_r0 + 1 >= height:
                        continue
                    source_cells = _cells_from_bbox_and_missing(source_r0, source_c0, missing)
                    moved_cells = _move_cells(source_cells, dr, dc)
                    candidates.append(
                        {
                            "missing": missing,
                            "dr": dr,
                            "dc": dc,
                            "target_col": target_col,
                            "relation": relation,
                            "source_cells": source_cells,
                            "moved_cells": moved_cells,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _missing_requirements(object_count: int, forced_missing: tuple[int, int] | None) -> set[tuple[int, int]]:
    order = list(ALL_CORNERS)
    random.shuffle(order)
    required = set(order[: min(4, object_count)])
    if forced_missing is not None:
        required.add(forced_missing)
    while len(required) > object_count:
        required.remove(next(iter(required - {forced_missing})))
    return required


def _build_records(
    candidates: list[dict],
    control_cols: list[int],
    object_count: int,
    changed_target: int,
    changed_missing: tuple[int, int],
) -> list[dict] | None:
    required_missing = _missing_requirements(object_count, changed_missing)
    chosen: list[dict] = []
    source_blocked: set[tuple[int, int]] = set()
    moved_blocked: set[tuple[int, int]] = set()
    used_controls: set[int] = set()
    relations: set[str] = set()
    used_missing: set[tuple[int, int]] = set()

    def conflicts(candidate: dict) -> bool:
        return any(cell in source_blocked for cell in candidate["source_cells"]) or any(
            cell in moved_blocked for cell in candidate["moved_cells"]
        )

    def candidate_score(candidate: dict) -> tuple[int, int, int, float]:
        score = 0
        if candidate["target_col"] not in used_controls:
            score += 5
        if candidate["missing"] in (required_missing - used_missing):
            score += 4
        if "different" not in relations and candidate["relation"] == "different":
            score += 6
        if not (relations & {"same", "none"}) and candidate["relation"] in {"same", "none"}:
            score += 6
        if candidate["target_col"] == changed_target and candidate["missing"] == changed_missing:
            score += 7
        return (
            score,
            -abs(candidate["target_col"] - changed_target),
            1 if candidate["relation"] == "different" else 0,
            random.random(),
        )

    def can_still_finish(start_idx: int) -> bool:
        remaining = object_count - len(chosen)
        if remaining < 0:
            return False
        need_controls = set(control_cols) - used_controls
        need_missing = required_missing - used_missing
        need_diff = {"different"} - relations
        need_non_diff = set() if relations & {"same", "none"} else {"same_or_none"}
        if len(need_controls) > remaining or len(need_missing) > remaining:
            return False

        future = [cand for cand in candidates[start_idx:] if not conflicts(cand)]
        future_controls = {cand["target_col"] for cand in future}
        future_missing = {cand["missing"] for cand in future}
        future_relations = {cand["relation"] for cand in future}
        if not need_controls <= future_controls:
            return False
        if not need_missing <= future_missing:
            return False
        if need_diff and "different" not in future_relations:
            return False
        if need_non_diff and not (future_relations & {"same", "none"}):
            return False
        if not any(
            cand["target_col"] == changed_target
            and cand["missing"] == changed_missing
            and cand["relation"] == "different"
            and not conflicts(cand)
            for cand in future
        ):
            return False
        return True

    forced_done = False

    def backtrack(start_idx: int) -> bool:
        nonlocal forced_done, source_blocked, moved_blocked
        if len(chosen) == object_count:
            return (
                used_controls == set(control_cols)
                and required_missing <= used_missing
                and "different" in relations
                and bool(relations & {"same", "none"})
                and forced_done
            )
        if not can_still_finish(start_idx):
            return False

        pool = [
            (idx, cand)
            for idx, cand in enumerate(candidates[start_idx:], start=start_idx)
            if not conflicts(cand)
        ]
        pool.sort(key=lambda item: candidate_score(item[1]), reverse=True)

        for idx, cand in pool:
            must_force = not forced_done
            if must_force and not (
                cand["target_col"] == changed_target
                and cand["missing"] == changed_missing
                and cand["relation"] == "different"
            ):
                continue

            chosen.append(cand)
            old_source = source_blocked.copy()
            old_moved = moved_blocked.copy()
            old_relations = relations.copy()
            old_controls = used_controls.copy()
            old_missing = used_missing.copy()
            old_forced = forced_done

            source_blocked |= _expand8(cand["source_cells"], 30, 30)
            moved_blocked |= _expand8(cand["moved_cells"], 30, 30)
            relations.add(cand["relation"])
            used_controls.add(cand["target_col"])
            used_missing.add(cand["missing"])
            if (
                cand["target_col"] == changed_target
                and cand["missing"] == changed_missing
                and cand["relation"] == "different"
            ):
                forced_done = True

            if backtrack(idx + 1):
                return True

            chosen.pop()
            source_blocked = old_source
            moved_blocked = old_moved
            relations.clear()
            relations.update(old_relations)
            used_controls.clear()
            used_controls.update(old_controls)
            used_missing.clear()
            used_missing.update(old_missing)
            forced_done = old_forced

        if not forced_done:
            return False
        return False

    return chosen if backtrack(0) else None


def _render_example(
    height: int,
    width: int,
    control_cols: list[int],
    control_colors: dict[int, int],
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)
    for c in control_cols:
        inp[0, c] = control_colors[c]
        out[0, c] = control_colors[c]
    for record in records:
        for r, c in record["source_cells"]:
            inp[r, c] = GREY
        for r, c in record["moved_cells"]:
            out[r, c] = control_colors[record["target_col"]]
    return inp, out


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        info = _analyze_grid(inp, require_generator_checks=True)
    except Exception:
        return False
    rebuilt = np.full_like(inp, info["bg"])
    for c, color in info["controls"].items():
        rebuilt[0, c] = color
    relation_kinds = set()
    for record in info["records"]:
        relation_kinds.add(record["relation"])
        for r, c in record["moved_coords"]:
            rebuilt[r, c] = record["target_color"]
    return np.array_equal(rebuilt, out) and "different" in relation_kinds and bool(
        relation_kinds & {"same", "none"}
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    control_count = random.randint(3, 5)
    object_count = random.randint(control_count, 5)

    control_cols = _choose_control_columns(width, control_count)
    if control_cols is None:
        return None
    control_palette = random.sample(CONTROL_COLORS, control_count)
    control_colors = {c: color for c, color in zip(control_cols, control_palette, strict=False)}

    gap2_pairs = [(a, b) for a, b in zip(control_cols, control_cols[1:], strict=False) if b - a == 2]
    if not gap2_pairs:
        return None
    left_col, right_col = random.choice(gap2_pairs)
    if random.random() < 0.5:
        changed_target = left_col
        changed_missing = random.choice(((0, 0), (1, 0)))
    else:
        changed_target = right_col
        changed_missing = random.choice(((0, 1), (1, 1)))

    candidates = _generate_candidates(height, width, control_cols)
    records = _build_records(candidates, control_cols, object_count, changed_target, changed_missing)
    if records is None:
        return None

    inp, expected = _render_example(height, width, control_cols, control_colors, records)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
