import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


_BLACK = 0


def _label_gray_components(grid: np.ndarray, gray_color: int) -> np.ndarray:
    """4) Label connected gray regions (4-neighborhood).

    This helper uses an iterative BFS and returns a matrix where each gray pixel has its
    component id and non-gray pixels are -1.
    """

    h, w = grid.shape
    labels = -np.ones((h, w), dtype=int)
    next_label = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != gray_color or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label

            while q:
                cr, cc = q.popleft()
                nr1, nc1 = cr - 1, cc
                nr2, nc2 = cr + 1, cc
                nr3, nc3 = cr, cc - 1
                nr4, nc4 = cr, cc + 1

                for nr, nc in ((nr1, nc1), (nr2, nc2), (nr3, nc3), (nr4, nc4)):
                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue
                    if grid[nr, nc] != gray_color:
                        continue
                    if labels[nr, nc] != -1:
                        continue
                    labels[nr, nc] = next_label
                    q.append((nr, nc))

            next_label += 1

    return labels


def _infer_gray_color(grid: np.ndarray, corners: Sequence[int]) -> int:
    """3) Infer the gray color by frequency excluding black and obvious corner markers.

    In ARC-like tasks the gray background region is usually the dominant non-black color,
    while corner markers are only four single pixels.
    """

    h, w = grid.shape
    values = grid.ravel()
    values = values[values != _BLACK]
    if values.size == 0:
        return 5

    counts = np.bincount(values.astype(np.int64))
    forbidden = set(int(c) for c in corners)
    best = -1
    best_count = -1
    for color, cnt in enumerate(counts):
        if color == _BLACK:
            continue
        if color in forbidden:
            continue
        if cnt > best_count:
            best_count = cnt
            best = color

    if best >= 0:
        return best
    return int(np.max(values))


def _collect_seed_like_cells(
    grid: np.ndarray,
    gray_color: int,
    corner_colors: Sequence[int],
    labels: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """2) Separate seed cells and marker cells by gray-component adjacency.

    A seed is expected to touch exactly one gray component (inside one region).
    A boundary marker commonly touches multiple gray components or none.
    """

    h, w = grid.shape
    seeds: List[Tuple[int, int]] = []
    markers: List[Tuple[int, int]] = []

    corner_positions = {
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
    }
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == _BLACK or color == gray_color:
                continue
            if (r, c) in corner_positions:
                continue

            neigh_labels = set()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w:
                    lbl = int(labels[nr, nc])
                    if lbl >= 0:
                        neigh_labels.add(lbl)

            # If it is adjacent to exactly one gray region, treat as a seed.
            # Otherwise keep as arbitration marker candidate.
            if len(neigh_labels) == 1:
                seeds.append((r, c))
            else:
                markers.append((r, c))

    return seeds, markers


@dataclass
class _State:
    r: int
    c: int
    color: int
    dr: int
    dc: int
    alive: bool = True


def _is_corner(h: int, w: int, r: int, c: int) -> bool:
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    1. Decode growth rules:
       corner pixel colors define direction codes (top-left->right, top-right->down,
       bottom-right->left, bottom-left->up), then assign each seed that owns such color.

    2. Grow each seed one-dimensional stripe:
       each active seed repeatedly advances only along its direction through gray cells.

    3. Resolve conflicts / arbitration:
       - gray is paintable by one seed per step,
       - marker cells are paintable only by matching seed color,
       - when a marker is reached by its matching color, it becomes paintable and
         growth continues across it.

    4. Keep corner markers unchanged and return the finished numpy integer grid.
    """

    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D ARC grid")

    h, w = g.shape
    out = g.copy()
    if h < 2 or w < 2:
        return out

    corners = (int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, w - 1]), int(g[h - 1, 0]))
    direction_by_color: Dict[int, Tuple[int, int]] = {
        corners[0]: (0, 1),
        corners[1]: (1, 0),
        corners[2]: (0, -1),
        corners[3]: (-1, 0),
    }

    gray = _infer_gray_color(g, corners)
    labels = _label_gray_components(g, gray)
    seeds, _ = _collect_seed_like_cells(g, gray, corners, labels)

    states: List[_State] = []
    for r, c in seeds:
        color = int(g[r, c])
        direction = direction_by_color.get(color)
        if direction is None:
            continue
        states.append(_State(r=r, c=c, color=color, dr=direction[0], dc=direction[1]))

    if not states:
        return out

    alive_mask = np.ones(len(states), dtype=bool)

    while True:
        proposals: Dict[Tuple[int, int], List[int]] = {}
        attempts: Dict[int, Tuple[int, int]] = {}
        any_attempt = False

        for i, st in enumerate(states):
            if not alive_mask[i]:
                continue

            nr, nc = st.r + st.dr, st.c + st.dc
            if not (0 <= nr < h and 0 <= nc < w):
                alive_mask[i] = False
                continue
            if _is_corner(h, w, nr, nc):
                alive_mask[i] = False
                continue

            target_color = int(g[nr, nc])
            current_color = out[nr, nc]

            # Can move only through gray or through a matching arbitration marker.
            if target_color == _BLACK:
                # This branch can be reopened only by marker logic above.
                alive_mask[i] = False
                continue
            if target_color == gray:
                # Unpainted gray cell: potentially claim it.
                pass
            elif target_color == st.color:
                # Matching color means this pixel may be arbitration marker.
                # It is treated as passable and will be overwritten by this color.
                pass
            else:
                # Occupied by another color (including other seed or other marker)
                alive_mask[i] = False
                continue

            if current_color != target_color and current_color != st.color and current_color != gray:
                # Already filled by prior winner in this same step.
                alive_mask[i] = False
                continue

            attempts[i] = (nr, nc)
            proposals.setdefault((nr, nc), []).append(i)
            any_attempt = True

        if not any_attempt:
            break

        winner_by_state = np.zeros(len(states), dtype=bool)
        for (nr, nc), idxs in proposals.items():
            if len(idxs) == 1:
                i = idxs[0]
                winner_by_state[i] = True
                continue

            tcolor = int(g[nr, nc])
            if tcolor == gray:
                # Simultaneous claims on gray; keep deterministic single winner.
                i = min(idxs)
                winner_by_state[i] = True
                continue

            # Arbitration marker: only matching seed color can occupy.
            matching = [i for i in idxs if states[i].color == tcolor]
            if len(matching) == 1:
                winner_by_state[matching[0]] = True
            else:
                # no direct pass if multiple same-color claims; deterministic hold:
                # let the lowest color-indexed winner continue.
                i = min(idxs, key=lambda x: states[x].color)
                winner_by_state[i] = True

        progressed = False
        for i, st in enumerate(states):
            if not alive_mask[i]:
                continue

            if i not in attempts or not winner_by_state[i]:
                alive_mask[i] = False
                continue

            nr, nc = attempts[i]
            if _is_corner(h, w, nr, nc):
                alive_mask[i] = False
                continue

            if out[nr, nc] not in (gray, st.color):
                # Another color consumed it before applying.
                alive_mask[i] = False
                continue

            out[nr, nc] = st.color
            st.r, st.c = nr, nc
            progressed = True

        if not progressed:
            break

    return out


def _inside_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_isolated_cell(blocked: np.ndarray, r: int, c: int, min_margin: int = 1) -> bool:
    h, w = blocked.shape
    if not _inside_bounds(h, w, r, c):
        return False
    if blocked[r, c]:
        return False
    for dr in range(-min_margin, min_margin + 1):
        for dc in range(-min_margin, min_margin + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and blocked[rr, cc]:
                return False
    return True


def _grow_blob(rng: np.random.Generator, blocked: np.ndarray, start: Tuple[int, int], target_size: int) -> List[Tuple[int, int]]:
    h, w = blocked.shape
    if not _inside_bounds(h, w, *start):
        return []
    if blocked[start]:
        return []

    blob: List[Tuple[int, int]] = [start]
    frontier = deque([start])
    blocked[start] = True

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    safety = 0
    while len(blob) < target_size and frontier:
        safety += 1
        if safety > target_size * 20:
            break

        r, c = frontier[rng.integers(0, len(frontier))]
        dr, dc = dirs[rng.integers(0, 4)]
        nr, nc = r + dr, c + dc

        if not _inside_bounds(h, w, nr, nc) or blocked[nr, nc]:
            continue

        # Keep the blob from accidentally touching blocked areas to preserve separate regions.
        if not _is_isolated_cell(blocked, nr, nc, min_margin=1):
            continue

        blob.append((nr, nc))
        frontier.append((nr, nc))
        blocked[nr, nc] = True

        # Small branching chance keeps shapes irregular.
        if rng.random() < 0.02 and r != start[0] or c != start[1]:
            continue

    if len(blob) < max(4, target_size // 6):
        # rollback: keep empty if growth failed
        for rr, cc in blob:
            blocked[rr, cc] = False
        return []

    for rr, cc in blob:
        blocked[rr, cc] = False

    return blob


def generate_input() -> np.ndarray:
    """
    1) Randomly choose a canvas up to 30x30 with black background.

    2) Paint corner direction markers (distinct colors):
       left-top, right-top, right-bottom, left-bottom encode
       right, down, left, up respectively.

    3) Build several disconnected gray regions with 1-2 seeds each (unique colors).

    4) Place a few arbitration markers near random region boundaries.
       The marker color is copied from an existing seed.
    """

    rng = np.random.default_rng()
    h = int(rng.integers(12, 28))
    w = int(rng.integers(12, 28))
    grid = np.zeros((h, w), dtype=int)

    color_bank = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    gray_color = int(rng.choice(color_bank))

    # Exclude gray for corner markers.
    corner_pool = [c for c in color_bank if c != gray_color]
    tl, tr, br, bl = rng.choice(corner_pool, size=4, replace=False)
    grid[0, 0] = tl
    grid[0, w - 1] = tr
    grid[h - 1, w - 1] = br
    grid[h - 1, 0] = bl

    num_regions = int(rng.integers(2, 5))
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, w - 1] = True

    regions: List[List[Tuple[int, int]]] = []
    all_gray = set()
    seed_colors_used: List[int] = [tl, tr, br, bl]
    all_seed_positions: List[Tuple[int, int, int]] = []
    region_seed_colors: List[List[int]] = []

    # Build disjoint irregular gray components.
    attempts = 0
    while len(regions) < num_regions and attempts < 80:
        attempts += 1

        # Some regions are created near previous regions to make possible arbitration candidates.
        if len(regions) > 0 and rng.random() < 0.45:
            base = rng.integers(0, len(regions))
            base_cells = regions[base]
            if not base_cells:
                continue
            br_, bc_ = base_cells[rng.integers(0, len(base_cells))]
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][rng.integers(0, 4)]
            start = (br_ + 2 * dr, bc_ + 2 * dc)
        else:
            start = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))

        target_size = int(rng.integers(8, 26))
        candidate = _grow_blob(rng, blocked.copy(), start, target_size)
        if len(candidate) < 8:
            continue

        # Avoid adjacency with existing gray cells and also avoid touching corner markers.
        valid = True
        for r, c in candidate:
            if blocked[r, c]:
                valid = False
                break
            for nr, nc in _neighbors(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] == gray_color:
                        valid = False
                        break
            if not valid:
                break

        if not valid:
            continue

        for r, c in candidate:
            grid[r, c] = gray_color
            blocked[r, c] = True
            all_gray.add((r, c))
        regions.append(candidate)

    if not regions:
        # Small fallback to make sure the puzzle is valid.
        for i in range(2, h - 2):
            for j in range(2, w - 2):
                grid[i, j] = gray_color
        regions = [[(i, j) for i in range(2, h - 2) for j in range(2, w - 2)]]

    # Place seeds.
    available_colors = [c for c in color_pool_for_seed() if c != gray_color]

    for rid, region in enumerate(regions):
        region_seed_colors.append([])
        if not region:
            continue
        seed_budget = 1 if len(region) < 14 else int(rng.integers(1, 3))
        seed_budget = min(seed_budget, len(region), 2)

        for _ in range(seed_budget):
            if not available_colors:
                break
            # Prefer unused colors for seed diversity.
            color = int(rng.choice(available_colors))
            available_colors.remove(color)
            seed_colors_used.append(color)

            r, c = region[rng.integers(0, len(region))]
            while grid[r, c] == color:
                r, c = region[rng.integers(0, len(region))]
            grid[r, c] = color
            region_seed_colors[rid].append(color)
            all_seed_positions.append((r, c, color))

    # Add arbitration markers at boundaries where a one-cell black gap exists.
    # We purposely need marker color to match an existing seed.
    if len(all_seed_positions) >= 1:
        labels = _label_gray_components(grid, gray_color)

        marker_positions = []
        for rid, region in enumerate(regions):
            if not region:
                continue
            for rr, cc in region:
                for dr, dc in _NEIGHBOR_OFFSETS():
                    m_r, m_c = rr + dr, cc + dc
                    n_r, n_c = rr + 2 * dr, cc + 2 * dc
                    if not _inside_bounds(h, w, m_r, m_c) or not _inside_bounds(h, w, n_r, n_c):
                        continue
                    if grid[m_r, m_c] != 0:
                        continue
                    if labels[n_r, n_c] < 0:
                        # only black boundary + another gray component after it
                        continue

                    r_other = labels[n_r, n_c]
                    if r_other == labels[rr, cc]:
                        continue

                    src_colors = region_seed_colors[rid]
                    if not src_colors:
                        continue
                    color = int(src_colors[rng.integers(0, len(src_colors))])
                    if rng.random() < 0.55:
                        marker_positions.append((m_r, m_c, color))

        for mr, mc, color in marker_positions[: max(1, len(regions) // 2)]:
            grid[mr, mc] = color

    return grid


def color_pool_for_seed() -> List[int]:
    return [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _NEIGHBOR_OFFSETS() -> List[Tuple[int, int]]:
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


if __name__ == "__main__":
    # Quick self-check + compile-time sanity.
    import traceback

    sample = generate_input()
    out = main(sample)
    assert sample.shape == out.shape, "shape must be preserved"
    assert np.array_equal(sample[0, 0], out[0, 0])
    assert np.array_equal(sample[0, -1], out[0, -1])
    assert np.array_equal(sample[-1, 0], out[-1, 0])
    assert np.array_equal(sample[-1, -1], out[-1, -1])

    for _ in range(5):
        inp = generate_input()
        res = main(inp)
        assert inp.shape == res.shape
        assert np.array_equal(inp[0, 0], res[0, 0])
        assert np.array_equal(inp[0, -1], res[0, -1])
        assert np.array_equal(inp[-1, 0], res[-1, 0])
        assert np.array_equal(inp[-1, -1], res[-1, -1])
        if inp.size == 0:
            continue
        assert set(np.unique(res)).issubset(set(range(10)))

    print("self-check passed", sample.shape)
