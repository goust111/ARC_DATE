from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Container:
    top: int
    bottom: int
    left: int
    right: int
    hole_side: str | None
    hole_row: int | None

    @property
    def complete(self) -> bool:
        return self.hole_side is None

    def spans(self, col: int) -> bool:
        return self.left <= col <= self.right

    def interior_cols(self) -> list[int]:
        return list(range(self.left + 1, self.right))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_containers(grid: np.ndarray) -> list[Container]:
    h, w = grid.shape
    containers: list[Container] = []
    for top in range(h - 2):
        for left in range(w - 3):
            if grid[top, left] != WALL:
                continue
            for right in range(left + 3, w):
                if grid[top, right] != WALL:
                    continue
                if top > 0 and (grid[top - 1, left] == WALL or grid[top - 1, right] == WALL):
                    continue
                for bottom in range(top + 2, h):
                    patch = grid[top : bottom + 1, left : right + 1]
                    ph, pw = patch.shape
                    if np.any((patch != BG) & (patch != WALL)):
                        continue
                    if not np.all(patch[-1, :] == WALL):
                        continue
                    if not np.all(patch[:-1, 1:-1] == BG):
                        continue
                    left_missing = [i for i in range(ph - 1) if patch[i, 0] != WALL]
                    right_missing = [i for i in range(ph - 1) if patch[i, -1] != WALL]
                    if any(i == 0 for i in left_missing + right_missing):
                        continue
                    if len(left_missing) + len(right_missing) > 1:
                        continue
                    expected = set()
                    for rr in range(top, bottom):
                        if rr != top + (left_missing[0] if left_missing else -999):
                            expected.add((rr, left))
                        if rr != top + (right_missing[0] if right_missing else -999):
                            expected.add((rr, right))
                    for cc in range(left, right + 1):
                        expected.add((bottom, cc))
                    actual = {
                        (top + rr, left + cc)
                        for rr in range(ph)
                        for cc in range(pw)
                        if patch[rr, cc] == WALL
                    }
                    if actual != expected:
                        continue
                    hole_side: str | None = None
                    hole_row: int | None = None
                    if left_missing:
                        hole_side = "left"
                        hole_row = top + left_missing[0]
                    elif right_missing:
                        hole_side = "right"
                        hole_row = top + right_missing[0]
                    containers.append(Container(top, bottom, left, right, hole_side, hole_row))
                    break
    containers.sort(key=lambda box: (box.top, box.left, box.bottom, box.right))
    return containers


def _find_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) not in (BG, WALL)]
    markers.sort()
    return markers


def _fill_container(out: np.ndarray, container: Container, color: int) -> None:
    patch = out[container.top : container.bottom, container.left + 1 : container.right]
    patch[patch == BG] = color


def _next_event(
    containers: list[Container],
    captured_ids: set[int],
    row: int,
    col: int,
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for idx, container in enumerate(containers):
        if not container.spans(col):
            continue
        if idx in captured_ids:
            if row >= container.top:
                continue
            event_row = container.top
        else:
            event_row = container.top if col in (container.left, container.right) else container.bottom
            if row >= event_row:
                continue
        cand = (event_row, idx)
        if best is None or cand < best:
            best = cand
        elif best is not None and cand[0] == best[0] and cand[1] != best[1]:
            raise AssertionError("ambiguous fall path: two containers collide on the same event row")
    return best


def _simulate(
    grid: np.ndarray,
    markers: list[tuple[int, int]] | None = None,
    containers: list[Container] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if containers is None:
        containers = _find_containers(g)
    if markers is None:
        markers = _find_markers(g)

    out = g.copy()
    captured_ids: set[int] = set()
    histories: list[dict[str, object]] = []

    for col, color in markers:
        out[0, col] = BG
        state_seen: set[tuple[int, int, tuple[int, ...]]] = set()
        row = 1
        history: dict[str, object] = {
            "column": col,
            "color": color,
            "leak_count": 0,
            "hit_ids": [],
        }
        while True:
            state = (row, col, tuple(sorted(captured_ids)))
            if state in state_seen:
                raise AssertionError("cyclic leak routing is not allowed")
            state_seen.add(state)

            event = _next_event(containers, captured_ids, row, col)
            if event is None:
                out[-1, :] = color
                history["outcome"] = "bottom"
                break

            _event_row, idx = event
            history["hit_ids"].append(idx)
            container = containers[idx]

            if idx in captured_ids:
                history["outcome"] = "blocked"
                history["blocked_by"] = idx
                break

            if container.complete:
                _fill_container(out, container, color)
                captured_ids.add(idx)
                history["outcome"] = "captured"
                history["captured_id"] = idx
                break

            history["leak_count"] = int(history["leak_count"]) + 1
            assert container.hole_row is not None and container.hole_side is not None
            row = container.hole_row
            col = container.left - 1 if container.hole_side == "left" else container.right + 1
            assert 0 <= col < w, "leak must stay inside the grid"

        histories.append(history)

    return out, histories


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top-row marker through gray U-containers, filling a closed cavity or spilling onward through a wall hole.

    Input structure:
    - The grid uses black background, gray U-shaped containers, and 2 to 3 colored singleton markers on the top row.
    - Every container has two vertical side walls and one bottom wall; some side walls are complete and some have exactly one missing pixel.
    - Marker columns are distinct and each starts above at least one container span.

    Rule:
    - Erase the original marker pixels, then process the markers from left to right.
    - Simulate each marker falling vertically: an intact container catches it and fills the container interior with that marker color, while a leaky container ejects it from the hole side to the adjacent outside column and the fall continues from the hole row.
    - If a marker reaches the bottom without being caught, fill the entire bottom row with its color; if a later marker would enter a container that was already filled by an earlier marker, that filled container blocks it at its top edge and nothing new is painted.

    Key insight:
    - The missing wall pixel is not just a defect: it reroutes the marker into a different vertical lane, which changes which later container, if any, becomes the true destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    markers = _find_markers(g)
    containers = _find_containers(g)

    assert 2 <= len(markers) <= 3, "expected 2 or 3 top-row markers"
    assert 3 <= len(containers) <= 5, "expected 3 to 5 containers"
    assert all(grid[0, c] != WALL for c, _ in markers)
    assert len({c for c, _ in markers}) == len(markers)
    assert any(container.complete for container in containers)
    assert any(not container.complete for container in containers)
    for col, _color in markers:
        assert any(container.spans(col) for container in containers), "every marker must start above some container"

    out, _histories = _simulate(g, markers=markers, containers=containers)
    return out


def _paint_container(grid: np.ndarray, container: Container) -> None:
    for rr in range(container.top, container.bottom):
        if container.hole_side != "left" or rr != container.hole_row:
            grid[rr, container.left] = WALL
        if container.hole_side != "right" or rr != container.hole_row:
            grid[rr, container.right] = WALL
    grid[container.bottom, container.left : container.right + 1] = WALL


def _boxes_touch(a: Container, b: Container, gap: int = 1) -> bool:
    return not (
        a.bottom + gap < b.top
        or b.bottom + gap < a.top
        or a.right + gap < b.left
        or b.right + gap < a.left
    )


def _new_container(
    top: int,
    left: int,
    height: int,
    width: int,
    hole_side: str | None = None,
    hole_row: int | None = None,
) -> Container:
    return Container(top, top + height - 1, left, left + width - 1, hole_side, hole_row)


def _sample_container(
    rng: random.Random,
    h: int,
    w: int,
    existing: list[Container],
    *,
    top_range: tuple[int, int],
    left_range: tuple[int, int] | None = None,
    height_range: tuple[int, int] = (4, 6),
    width_range: tuple[int, int] = (4, 7),
    must_span: int | None = None,
    avoid_span_cols: set[int] | None = None,
    hole_side: str | None = None,
) -> Container:
    avoid_span_cols = avoid_span_cols or set()
    for _ in range(220):
        height = rng.randint(*height_range)
        width = rng.randint(*width_range)
        top_lo, top_hi = top_range
        top_hi = min(top_hi, h - height - 2)
        if top_lo > top_hi:
            continue
        top = rng.randint(top_lo, top_hi)

        if left_range is None:
            left_lo, left_hi = 1, w - width - 2
        else:
            left_lo, left_hi = left_range
            left_hi = min(left_hi, w - width - 2)
        if left_lo > left_hi:
            continue
        left = rng.randint(left_lo, left_hi)
        cont = _new_container(top, left, height, width)

        if must_span is not None and not cont.spans(must_span):
            continue
        if any(cont.spans(col) for col in avoid_span_cols):
            continue
        if any(_boxes_touch(cont, other) for other in existing):
            continue

        if hole_side is not None:
            if hole_side == "left" and cont.left <= 0:
                continue
            if hole_side == "right" and cont.right >= w - 1:
                continue
            cont = Container(
                cont.top,
                cont.bottom,
                cont.left,
                cont.right,
                hole_side,
                rng.randint(cont.top + 1, cont.bottom - 1),
            )
        return cont
    raise AssertionError("failed constrained container placement")


def _random_unused_container(
    rng: random.Random,
    h: int,
    w: int,
    placed: list[Container],
    forbidden_cols: set[int],
) -> Container | None:
    for _ in range(120):
        height = rng.randint(4, 6)
        width = rng.randint(4, 7)
        top = rng.randint(2, h - height - 2)
        left = rng.randint(1, w - width - 2)
        cont = Container(top, top + height - 1, left, left + width - 1, None, None)
        if any(_boxes_touch(cont, other) for other in placed):
            continue
        if forbidden_cols & set(range(cont.left, cont.right + 1)):
            continue
        if rng.random() < 0.55:
            side = "left" if rng.random() < 0.5 else "right"
            if side == "left" and cont.left <= 0:
                side = "right"
            if side == "right" and cont.right >= w - 1:
                side = "left"
            cont = Container(
                cont.top,
                cont.bottom,
                cont.left,
                cont.right,
                side,
                rng.randint(cont.top + 1, cont.bottom - 1),
            )
        return cont
    return None


def _blocked_case(rng: random.Random) -> tuple[np.ndarray, str]:
    h = rng.randint(15, 20)
    w = rng.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    a = _sample_container(
        rng,
        h,
        w,
        [],
        top_range=(2, 4),
        left_range=(1, max(2, w // 5)),
        height_range=(4, 5),
        width_range=(4, 6),
        hole_side="right",
    )
    leak_col = a.right + 1

    b = _sample_container(
        rng,
        h,
        w,
        [a],
        top_range=(a.bottom + 2, h - 6),
        left_range=(max(1, leak_col - 2), min(leak_col, w - 6)),
        height_range=(4, 5),
        width_range=(5, 7),
        must_span=leak_col,
    )

    m1 = rng.choice(a.interior_cols())
    b_choices = [c0 for c0 in b.interior_cols() if c0 > a.right]
    if not b_choices:
        raise AssertionError("blocked case lacks direct target column")
    m2 = rng.choice(b_choices)
    c = _sample_container(
        rng,
        h,
        w,
        [a, b],
        top_range=(2, h - 6),
        height_range=(4, 5),
        width_range=(4, 6),
        avoid_span_cols=set(range(b.left, b.right + 1)) | {leak_col},
    )
    containers = [a, b, c]
    if any(_boxes_touch(x, y) for x, y in combinations(containers, 2)):
        raise AssertionError("blocked case containers touch")

    grid.fill(BG)
    for cont in containers:
        _paint_container(grid, cont)

    markers = [(m1, None), (m2, None)]
    if rng.random() < 0.5:
        extra_cols = [c0 for c0 in c.interior_cols() if c0 not in {m1, m2}]
        if extra_cols:
            markers.append((rng.choice(extra_cols), None))

    return _finalize_case(rng, grid, containers, [c for c, _ in markers], {leak_col}), "blocked"


def _bottom_case(rng: random.Random) -> tuple[np.ndarray, str]:
    h = rng.randint(15, 20)
    w = rng.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    side = "right"
    a = _sample_container(
        rng,
        h,
        w,
        [],
        top_range=(2, 4),
        left_range=(1, max(2, w // 4)),
        height_range=(4, 5),
        width_range=(4, 6),
        hole_side=side,
    )
    leak_col = a.left - 1 if side == "left" else a.right + 1

    b = _sample_container(
        rng,
        h,
        w,
        [a],
        top_range=(7, h - 6),
        left_range=(max(leak_col + 3, w // 2), w - 6),
        height_range=(4, 6),
        width_range=(5, 7),
        avoid_span_cols={leak_col},
    )

    c = _sample_container(
        rng,
        h,
        w,
        [a, b],
        top_range=(2, h - 6),
        height_range=(4, 5),
        width_range=(4, 6),
        avoid_span_cols={leak_col},
    )

    containers = [a, b, c]
    if any(_boxes_touch(x, y) for x, y in combinations(containers, 2)):
        raise AssertionError("bottom case containers touch")

    for cont in containers:
        _paint_container(grid, cont)

    m1 = rng.choice(a.interior_cols())
    m2 = rng.choice(b.interior_cols())
    marker_cols = sorted({m1, m2})
    if len(marker_cols) != 2:
        raise AssertionError("bottom case duplicate marker columns")
    if rng.random() < 0.5:
        extra = [c0 for c0 in c.interior_cols() if c0 not in marker_cols]
        if extra:
            marker_cols.append(rng.choice(extra))
            marker_cols.sort()

    return _finalize_case(rng, grid, containers, marker_cols, {leak_col}), "bottom"


def _multileak_case(rng: random.Random) -> tuple[np.ndarray, str]:
    h = rng.randint(16, 20)
    w = rng.randint(19, 24)
    grid = np.zeros((h, w), dtype=int)

    a = _sample_container(
        rng,
        h,
        w,
        [],
        top_range=(2, 4),
        left_range=(1, max(2, w // 5)),
        height_range=(4, 5),
        width_range=(4, 6),
        hole_side="right",
    )
    leak1 = a.right + 1

    b = _sample_container(
        rng,
        h,
        w,
        [a],
        top_range=(a.bottom + 2, h - 8),
        left_range=(max(1, leak1 - 1), min(leak1, w - 6)),
        height_range=(4, 5),
        width_range=(4, 6),
        must_span=leak1,
        hole_side="right",
    )
    leak2 = b.right + 1

    c = _sample_container(
        rng,
        h,
        w,
        [a, b],
        top_range=(b.bottom + 2, h - 6),
        left_range=(max(1, leak2 - 2), min(leak2, w - 7)),
        height_range=(4, 6),
        width_range=(5, 7),
        must_span=leak2,
    )

    containers = [a, b, c]
    if any(_boxes_touch(x, y) for x, y in combinations(containers, 2)):
        raise AssertionError("multileak case containers touch")

    for cont in containers:
        _paint_container(grid, cont)

    m1 = rng.choice(a.interior_cols())
    direct_cols = [c0 for c0 in c.interior_cols() if c0 > b.right]
    if not direct_cols:
        raise AssertionError("multileak case lacks a direct blocked column")
    m2 = rng.choice(direct_cols)
    marker_cols = sorted({m1, m2})
    if len(marker_cols) != 2:
        raise AssertionError("multileak case duplicate marker columns")
    if rng.random() < 0.45:
        extra = [c0 for c0 in c.interior_cols() if c0 not in marker_cols and c0 > marker_cols[0]]
        if extra:
            marker_cols.append(rng.choice(extra))
            marker_cols.sort()

    return _finalize_case(rng, grid, containers, marker_cols, {leak1, leak2}), "multileak"


def _finalize_case(
    rng: random.Random,
    grid: np.ndarray,
    containers: list[Container],
    marker_cols: list[int],
    protected_cols: set[int],
) -> np.ndarray:
    h, w = grid.shape
    used_cols = set(marker_cols) | set(protected_cols)
    placed = list(containers)
    extra_target = rng.randint(0, max(0, 5 - len(placed)))
    for _ in range(extra_target):
        extra = _random_unused_container(rng, h, w, placed, used_cols)
        if extra is None:
            break
        placed.append(extra)
        _paint_container(grid, extra)

    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    for col, color in zip(marker_cols, colors):
        grid[0, col] = color

    parsed = _find_containers(grid)
    if len(parsed) != len(placed):
        raise AssertionError("parser could not recover every generated container")
    return grid


def _history_signature(histories: list[dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    sig = []
    for hist in histories:
        sig.append(
            (
                hist["column"],
                hist["outcome"],
                hist["leak_count"],
                tuple(hist["hit_ids"]),
                hist.get("captured_id"),
                hist.get("blocked_by"),
            )
        )
    return tuple(sig)


def _validate_case(grid: np.ndarray, family: str) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.all(g[1:, :] != -1)

    markers = _find_markers(g)
    containers = _find_containers(g)
    assert 2 <= len(markers) <= 3
    assert 3 <= len(containers) <= 5
    assert any(container.complete for container in containers)
    assert any(not container.complete for container in containers)
    assert len({col for col, _ in markers}) == len(markers)
    for col, _ in markers:
        assert any(container.spans(col) for container in containers)

    out, histories = _simulate(g, markers=markers, containers=containers)
    assert np.array_equal(out, main(g))
    assert any(int(hist["leak_count"]) >= 1 for hist in histories), "at least one marker must use a leak"
    assert len(set(_history_signature(histories))) == len(histories), "marker histories should stay distinguishable"
    assert np.count_nonzero(out != g) > len(markers), "output must change beyond erasing the markers"

    if family == "blocked":
        assert any(hist["outcome"] == "blocked" for hist in histories)
        assert any(hist["outcome"] == "captured" for hist in histories)
    elif family == "bottom":
        assert any(hist["outcome"] == "bottom" for hist in histories)
        assert any(hist["outcome"] == "captured" for hist in histories)
    else:
        assert any(int(hist["leak_count"]) >= 2 for hist in histories)
        assert any(hist["outcome"] == "captured" for hist in histories)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        family = rng.choice(("blocked", "bottom", "multileak"))
        try:
            if family == "blocked":
                grid, family_name = _blocked_case(rng)
            elif family == "bottom":
                grid, family_name = _bottom_case(rng)
            else:
                grid, family_name = _multileak_case(rng)
            _validate_case(grid, family_name)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid gravity-routing container puzzle")
