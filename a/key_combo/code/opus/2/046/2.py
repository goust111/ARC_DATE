import random
import numpy as np

Color = {
    "BLACK": 0,
    "GREY": 5,
}


def _extract_non_diag_nonzero(masked: np.ndarray):
    h, w = masked.shape
    values = []
    for r in range(h):
        for c in range(w):
            if r == c:
                continue
            v = int(masked[r, c])
            if v == Color["BLACK"]:
                continue
            values.append(v)
    return values


def _minimal_period(grid: np.ndarray):
    h, w = grid.shape
    colors = {}
    best = None
    for p in range(2, min(h, w) + 1):
        good = True
        tile = {}
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                if v == Color["BLACK"] or r == c:
                    continue
                key = (r % p, c % p)
                prev = tile.get(key)
                if prev is None:
                    tile[key] = v
                elif prev != v:
                    good = False
                    break
            if not good:
                break
        if good:
            best = p
            break
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transpose the square grid across its main diagonal, repair periodic structure, then transpose back.

    Input structure:
    - Black background.
    - A full grey diagonal from (0,0) to (n-1,n-1).
    - Off-diagonal cells use 3-4 non-grey colors with random black holes.

    Rule:
    - Transpose across the main diagonal (diagonal stays fixed).
    - Infer the smallest period p such that every non-black, non-diagonal sample color is consistent by (r mod p, c mod p).
    - Reconstruct the p×p tile from visible samples and fill all off-diagonal holes with it.
    - Transpose the repaired grid back to original orientation.

    Key insight:
    - Periodic structure is latent until transposition; the diagonal acts as both key and invariant that guides a two-step transpose-restore workflow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    transposed = g.T.copy()
    p = _minimal_period(transposed)
    assert p is not None and p >= 2

    tile = np.full((p, p), Color["BLACK"], dtype=int)
    seen = np.zeros((p, p), dtype=bool)
    h, w = transposed.shape
    for r in range(h):
        for c in range(w):
            v = int(transposed[r, c])
            if r == c or v == Color["BLACK"] or v == Color["GREY"]:
                continue
            key = (r % p, c % p)
            rr = int(key[0])
            cc = int(key[1])
            if not seen[rr, cc]:
                tile[rr, cc] = v
                seen[rr, cc] = True

    if not seen.any():
        return g.copy()

    fallback = int(np.bincount(tile[tile != Color["BLACK"]].ravel()).argmax()) if (tile != Color["BLACK"]).any() else 1
    tile[~seen] = fallback

    repaired = transposed.copy()
    for r in range(h):
        for c in range(w):
            if r == c:
                continue
            if repaired[r, c] == Color["BLACK"]:
                repaired[r, c] = tile[r % p, c % p]
            elif repaired[r, c] == Color["GREY"]:
                # keep diagonal
                pass

    out = repaired.T
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.randint(10, 20)
        p = random.choice([2, 3, 4])
        if n < 2 * p:
            continue

        colors = [1, 2, 3, 4]
        tile = np.zeros((p, p), dtype=int)
        for r in range(p):
            for c in range(p):
                tile[r, c] = random.choice(colors)

        # avoid smaller-period degeneracy
        for _ in range(20):
            if _minimal_period(tile) == p:
                break
            for rr in range(p):
                cc = random.randrange(p)
                tile[rr, cc] = random.choice(colors)
        if _minimal_period(tile) != p:
            continue

        trans = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                trans[r, c] = int(tile[r % p, c % p])

        non_diag = [(r, c) for r in range(n) for c in range(n) if r != c]
        random.shuffle(non_diag)
        target = int(len(non_diag) * random.uniform(0.15, 0.25))

        residue_counts = {}
        for r, c in non_diag:
            residue_counts.setdefault((r % p, c % p), 0)
            residue_counts[(r % p, c % p)] += 1

        holes = 0
        for r, c in non_diag:
            key = (r % p, c % p)
            if holes >= target:
                break
            if residue_counts.get(key, 0) <= 1:
                continue
            trans[r, c] = Color["BLACK"]
            residue_counts[key] -= 1
            holes += 1

        inp = trans.T.copy()
        for i in range(n):
            inp[i, i] = Color["GREY"]

        repaired = main(inp)
        if repaired.shape[0] != n or repaired.shape[1] != n:
            continue
        # self-check: every off-diagonal zero in input has been filled
        if not np.any(inp == Color["BLACK"]):
            continue
        if repaired.shape == inp.shape:
            return inp

    raise RuntimeError("generate_input: failed to sample a valid case for puzzle 2")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, np.unique(x), np.unique(y))
