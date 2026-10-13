from __future__ import annotations

from collections import Counter
import itertools
import random
from typing import Iterable

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(sorted(coords), dtype=int)
            components.append(arr)
    components.sort(key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min())))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_winner(votes: Counter[int]) -> int | None:
    if not votes:
        return None
    top = max(votes.values())
    winners = [color for color, count in votes.items() if count == top]
    return int(winners[0]) if len(winners) == 1 else None


def _most_common_unique(counts: Counter[int], colors: Iterable[int], role: str) -> int:
    pool = [int(color) for color in colors]
    assert pool, f"expected at least one {role} color"
    top = max(int(counts[color]) for color in pool)
    winners = [color for color in pool if int(counts[color]) == top]
    assert len(winners) == 1, f"expected a unique {role} color"
    return int(winners[0])


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    counts = Counter(map(int, g.ravel()))
    colors = list(counts)
    bg = _most_common_unique(counts, colors, "background")
    wall = _most_common_unique(counts, [color for color in colors if color != bg], "wall")

    chambers = _components_mask(g == bg)
    assert chambers, "expected at least one background chamber"
    h, w = g.shape
    chamber_id = np.full((h, w), -1, dtype=int)
    for idx, coords in enumerate(chambers):
        chamber_id[coords[:, 0], coords[:, 1]] = int(idx)

    marker_colors = set(map(int, np.unique(g))) - {int(bg), int(wall)}
    base_votes = [Counter() for _ in chambers]
    for r, c in np.argwhere(np.isin(g, list(marker_colors))):
        rr = int(r)
        cc = int(c)
        touched = {
            int(chamber_id[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and chamber_id[nr, nc] >= 0
        }
        assert 1 <= len(touched) <= 2, "markers must touch one or two chambers"
        for idx in touched:
            base_votes[idx][int(g[rr, cc])] += 1

    adjacency = [set() for _ in chambers]
    for r, c in np.argwhere(g == wall):
        rr = int(r)
        cc = int(c)
        touched = {
            int(chamber_id[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and chamber_id[nr, nc] >= 0
        }
        for left, right in itertools.combinations(sorted(touched), 2):
            adjacency[left].add(right)
            adjacency[right].add(left)

    return {
        "grid": g,
        "bg": int(bg),
        "wall": int(wall),
        "chambers": chambers,
        "chamber_id": chamber_id,
        "base_votes": base_votes,
        "adjacency": adjacency,
    }


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    info = _analyze_grid(grid)
    g = info["grid"]
    chambers = info["chambers"]
    base_votes = [Counter(votes) for votes in info["base_votes"]]
    adjacency = info["adjacency"]

    out = g.copy()
    initial_only = g.copy()
    final_colors: list[int | None] = [None] * len(chambers)
    rounds: list[int | None] = [None] * len(chambers)
    unresolved = set(range(len(chambers)))
    round_idx = 0

    while unresolved:
        newly_colored: list[tuple[int, int]] = []
        for idx in sorted(unresolved):
            votes = Counter(base_votes[idx])
            for nbr in adjacency[idx]:
                fill = final_colors[nbr]
                if fill is not None:
                    votes[int(fill)] += 1
            winner = _unique_winner(votes)
            if winner is not None:
                newly_colored.append((idx, int(winner)))

        assert newly_colored, "vote propagation got stuck before all chambers were resolved"

        if round_idx == 0:
            for idx, fill in newly_colored:
                coords = chambers[idx]
                initial_only[coords[:, 0], coords[:, 1]] = int(fill)

        for idx, fill in newly_colored:
            final_colors[idx] = int(fill)
            rounds[idx] = int(round_idx)
            coords = chambers[idx]
            out[coords[:, 0], coords[:, 1]] = int(fill)
            unresolved.remove(idx)
        round_idx += 1

    return out, {
        **info,
        "initial_only": initial_only,
        "final_colors": [int(color) for color in final_colors],
        "rounds": [int(round_id) for round_id in rounds],
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill every chamber by marker-majority first, then resolve the tied chambers by neighbor votes cascading across shared walls.

    Input structure:
    - The grid has one dominant background color, one dominant wall color, and a
      few colored marker pixels embedded in the walls.
    - The background cells form 4-connected chambers separated by wall and marker
      cells; each marker touches one or two chambers orthogonally.

    Rule:
    - Infer the background as the unique most common color and the wall as the
      unique most common remaining color, then find the 4-connected background
      chambers.
    - Round 0: each chamber with a unique majority among its adjacent marker
      colors is filled with that majority color.
    - Later rounds: every unresolved chamber adds one vote from each already
      colored neighboring chamber, where neighbors are chamber pairs that share at
      least one orthogonally adjacent wall cell; if the combined votes now have a
      unique majority, fill that whole chamber with that color. Keep walls and
      markers unchanged.

    Key insight:
    - A tied chamber is not decided locally: solved chambers pass one vote each
      through the separating walls, so the order of chamber resolution is part of
      the puzzle's meaning.
    """
    out, _details = _solve_with_details(np.asarray(input_grid, dtype=int))
    return out


_NON_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_PATTERN_CACHE: dict[tuple[tuple[int, ...], int], list[tuple[str, int, tuple[int, ...]]]] = {}


def _enumerate_vote_patterns(colors: tuple[int, ...], max_len: int = 3) -> list[tuple[int, ...]]:
    patterns = [()]
    for length in range(1, max_len + 1):
        patterns.extend(itertools.combinations_with_replacement(colors, length))
    return patterns


def _choose_base_votes(
    target: int,
    neighbor_colors: list[int],
    vote_colors: tuple[int, ...],
    *,
    seed: bool,
    prefer: str,
) -> Counter[int] | None:
    key = (vote_colors, int(target))
    if key not in _PATTERN_CACHE:
        entries: list[tuple[str, int, tuple[int, ...]]] = []
        for pattern in _enumerate_vote_patterns(vote_colors, max_len=3):
            votes = Counter(pattern)
            total = sum(votes.values())
            kind = "blank" if total == 0 else ("major" if _unique_winner(votes) is not None else "tie")
            entries.append((kind, int(total), pattern))
        _PATTERN_CACHE[key] = entries

    neighbor_votes = Counter(map(int, neighbor_colors))
    candidates: list[tuple[str, int, tuple[int, ...]]] = []
    for kind, total, pattern in _PATTERN_CACHE[key]:
        votes = Counter(pattern)
        if seed:
            if _unique_winner(votes) != int(target):
                continue
        else:
            if _unique_winner(votes) is not None:
                continue
            if _unique_winner(votes + neighbor_votes) != int(target):
                continue
        candidates.append((kind, total, pattern))

    if not candidates:
        return None

    def sort_key(item: tuple[str, int, tuple[int, ...]]):
        kind, total, _pattern = item
        return (kind != prefer, total, random.random())

    candidates.sort(key=sort_key)
    return Counter(candidates[0][2])


def _choose_seed_layout(n_chambers: int, require_contested: bool) -> tuple[list[int], list[int]]:
    for _ in range(200):
        seed_count = random.choice([2, 2, 3])
        seeds = sorted(random.sample(range(n_chambers), seed_count))
        rounds = [min(abs(idx - seed) for seed in seeds) for idx in range(n_chambers)]
        if max(rounds) < 2:
            continue
        if require_contested:
            contested = any(
                rounds[idx] > 0
                and sum(
                    1
                    for nbr in (idx - 1, idx + 1)
                    if 0 <= nbr < n_chambers and rounds[nbr] == rounds[idx] - 1
                )
                == 2
                for idx in range(n_chambers)
            )
            if not contested:
                continue
        return seeds, rounds
    raise RuntimeError("failed to build a valid chamber propagation layout")


def _transform_coords(
    coords: np.ndarray,
    shape: tuple[int, int],
    *,
    transpose: bool,
    flip_ud: bool,
    flip_lr: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    arr = coords.copy()
    h, w = map(int, shape)
    if transpose:
        arr = arr[:, [1, 0]]
        h, w = w, h
    if flip_ud:
        arr[:, 0] = h - 1 - arr[:, 0]
    if flip_lr:
        arr[:, 1] = w - 1 - arr[:, 1]
    arr = np.array(sorted((int(r), int(c)) for r, c in arr), dtype=int)
    return arr, (h, w)


def _build_case() -> tuple[np.ndarray, dict]:
    want_blank = random.random() < 0.6
    want_shared = random.random() < 0.5
    want_contested = random.random() < 0.55

    n_chambers = random.randint(5, 7)
    seeds, rounds = _choose_seed_layout(n_chambers, require_contested=want_contested)

    role_colors = random.sample(_NON_BLACK, 6)
    bg = int(role_colors[0])
    wall = int(role_colors[1])
    vote_colors = tuple(int(color) for color in role_colors[2:6])
    target_pool = list(vote_colors[: random.choice([2, 3])])

    targets: list[int | None] = [None] * n_chambers
    base_votes: list[Counter[int] | None] = [None] * n_chambers

    seed_colors = random.sample(target_pool, min(len(target_pool), len(seeds)))
    while len(seed_colors) < len(seeds):
        seed_colors.append(random.choice(target_pool))
    random.shuffle(seed_colors)

    for idx, seed in enumerate(seeds):
        target = int(seed_colors[idx])
        targets[seed] = target
        base_votes[seed] = _choose_base_votes(target, [], vote_colors, seed=True, prefer="major")
        assert base_votes[seed] is not None

    for round_id in range(1, max(rounds) + 1):
        chamber_ids = [idx for idx, value in enumerate(rounds) if value == round_id]
        random.shuffle(chamber_ids)
        for idx in chamber_ids:
            neighbor_colors = [
                int(targets[nbr])
                for nbr in (idx - 1, idx + 1)
                if 0 <= nbr < n_chambers and rounds[nbr] == round_id - 1
            ]
            options = list(dict.fromkeys(neighbor_colors))
            random.shuffle(options)
            choice = None
            for target in options:
                prefer = "blank" if len(set(neighbor_colors)) == 1 and random.random() < 0.5 else "tie"
                votes = _choose_base_votes(int(target), neighbor_colors, vote_colors, seed=False, prefer=prefer)
                if votes is not None:
                    choice = (int(target), votes)
                    break
            assert choice is not None, "failed to choose marker votes for a propagated chamber"
            targets[idx], base_votes[idx] = choice

    final_targets = [int(target) for target in targets]
    final_base_votes = [Counter(votes) for votes in base_votes]
    assert len(set(final_targets)) >= 2
    if want_blank and not any(sum(votes.values()) == 0 for idx, votes in enumerate(final_base_votes) if rounds[idx] > 0):
        raise RuntimeError("wanted a no-marker chamber but did not get one")

    dims = None
    for _ in range(80):
        widths = [random.randint(3, 4) for _ in range(n_chambers)]
        chamber_h = random.randint(6, 8)
        total_w = sum(widths) + n_chambers + 1
        total_h = chamber_h + 2
        if max(total_h, total_w) > 30:
            continue
        bg_count = sum(widths) * chamber_h
        wall_count = 2 * total_w + (n_chambers + 1) * chamber_h
        if bg_count <= wall_count:
            continue
        dims = (widths, chamber_h, total_h, total_w)
        break
    if dims is None:
        raise RuntimeError("failed to sample a valid ARC-sized canvas")

    widths, chamber_h, total_h, total_w = dims
    grid = np.full((total_h, total_w), int(wall), dtype=int)
    chamber_coords: list[np.ndarray] = []
    starts: list[int] = []
    col = 1
    for width in widths:
        grid[1 : 1 + chamber_h, col : col + width] = int(bg)
        coords = np.array(
            [(r, c) for r in range(1, 1 + chamber_h) for c in range(col, col + width)],
            dtype=int,
        )
        chamber_coords.append(coords)
        starts.append(col)
        col += width + 1

    pending_votes = [Counter(votes) for votes in final_base_votes]
    shared_used = False
    for idx in range(n_chambers - 1):
        separator_col = starts[idx] + widths[idx]
        common = [
            color
            for color in list(pending_votes[idx])
            if pending_votes[idx][color] > 0 and pending_votes[idx + 1][color] > 0
        ]
        random.shuffle(common)
        if common and chamber_h >= 4 and random.random() < 0.5:
            row = random.randint(1, chamber_h)
            color = int(common[0])
            grid[row, separator_col] = color
            pending_votes[idx][color] -= 1
            pending_votes[idx + 1][color] -= 1
            if pending_votes[idx][color] == 0:
                del pending_votes[idx][color]
            if pending_votes[idx + 1][color] == 0:
                del pending_votes[idx + 1][color]
            shared_used = True

    if want_shared and not shared_used:
        raise RuntimeError("wanted a shared wall marker but did not place one")

    for idx in range(n_chambers):
        width = widths[idx]
        c0 = starts[idx]
        slots = [(0, c) for c in range(c0, c0 + width)] + [(total_h - 1, c) for c in range(c0, c0 + width)]
        if idx == 0:
            slots += [(r, 0) for r in range(1, 1 + chamber_h)]
        if idx == n_chambers - 1:
            slots += [(r, total_w - 1) for r in range(1, 1 + chamber_h)]
        random.shuffle(slots)
        needed = sum(pending_votes[idx].values())
        assert needed <= len(slots)
        slot_idx = 0
        for color, count in list(pending_votes[idx].items()):
            for _ in range(count):
                rr, cc = slots[slot_idx]
                slot_idx += 1
                grid[rr, cc] = int(color)

    transpose = random.random() < 0.5
    flip_ud = random.random() < 0.5
    flip_lr = random.random() < 0.5
    if transpose:
        grid = grid.T.copy()
    if flip_ud:
        grid = np.flipud(grid).copy()
    if flip_lr:
        grid = np.fliplr(grid).copy()

    transformed_chambers: list[np.ndarray] = []
    shape = (total_h, total_w)
    for coords in chamber_coords:
        arr, _new_shape = _transform_coords(coords, shape, transpose=transpose, flip_ud=flip_ud, flip_lr=flip_lr)
        transformed_chambers.append(arr)

    meta = {
        "bg": int(bg),
        "wall": int(wall),
        "targets": final_targets,
        "rounds": [int(value) for value in rounds],
        "base_votes": final_base_votes,
        "chambers": transformed_chambers,
        "wanted_blank": bool(want_blank),
        "wanted_shared": bool(want_shared),
        "wanted_contested": bool(want_contested),
    }
    _validate_case(grid, meta)
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> None:
    out, details = _solve_with_details(grid)
    assert isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= 30
    assert isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30

    chamber_lookup = {
        tuple(map(tuple, coords.tolist())): idx
        for idx, coords in enumerate(details["chambers"])
    }

    saw_blank = False
    for expected_idx, coords in enumerate(meta["chambers"]):
        key = tuple(map(tuple, coords.tolist()))
        assert key in chamber_lookup, "generated chamber geometry changed unexpectedly"
        actual_idx = chamber_lookup[key]
        expected_target = int(meta["targets"][expected_idx])
        expected_round = int(meta["rounds"][expected_idx])
        expected_votes = Counter(meta["base_votes"][expected_idx])

        assert details["final_colors"][actual_idx] == expected_target
        assert details["rounds"][actual_idx] == expected_round
        assert details["base_votes"][actual_idx] == expected_votes

        filled = out[coords[:, 0], coords[:, 1]]
        assert np.all(filled == expected_target)
        if sum(expected_votes.values()) == 0:
            saw_blank = True

    assert any(round_id > 0 for round_id in meta["rounds"])
    assert max(meta["rounds"]) >= 2
    assert not np.array_equal(details["initial_only"], out), "cascade step must matter"
    if meta["wanted_blank"]:
        assert saw_blank, "requested a no-marker chamber but none survived validation"


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, _meta = _build_case()
            return grid
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid chained-chamber ARC input")
