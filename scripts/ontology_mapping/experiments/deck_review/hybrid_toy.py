#!/usr/bin/env python3
"""
What the hybrid method does to the decision regions, computed on a toy example (deck slide
"The same toy points, with the boundaries actually computed"; report §4.6).

16 training samples of three labels A, B, C and one term vector per label, at the 2-D positions of the
deck's sketch. Each point (u, v) of the plane is embedded as the unit vector of (u, v, h), so that
cosines behave like on the sphere of real embeddings. Linear = RidgeClassifier (alpha 1) on the samples,
as methods.linear_scores; hybrid = ridge score + 2 * cos(x, t_c), as methods.hybrid. Every point of the
plane is coloured by its winning label, before and after.

Findings: each region grows toward its own term vector (A along the bottom near t_A, B along the top near
t_B, C along the right of the B / C boundary); 5 % of the plane changes label; the sketch's new sample x
stays A (scores A / B / C 0.12 / -0.43 / -0.69 -> 1.51 / 0.79 / 0.26). Boundaries bend slightly because the
cosine is not linear in the plane coordinates.

python experiments/deck_review/hybrid_toy.py --figure hybrid_toy.png
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import RidgeClassifier

A = [(120, 90), (180, 140), (90, 200), (220, 80), (160, 250), (250, 160)]
B = [(560, 110), (630, 160), (520, 70), (680, 90), (600, 220)]
C = [(480, 390), (560, 385), (620, 350), (530, 335), (600, 315)]
TERMS = {"A": (70, 330), "B": (700, 40), "C": (690, 410)}
X0 = (345, 165)  # the sketch's new sample


def embed(points, h):
    p = np.atleast_2d(np.asarray(points, float))
    e = np.c_[(p[:, 0] - 380) / 300, (p[:, 1] - 220) / 300, np.full(len(p), h)]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=float, default=1.0, help="height of the plane above the origin (sets the cosine spread)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--weight", type=float, default=2.0, help="cosine weight of the hybrid (2 in methods.hybrid)")
    ap.add_argument("--figure", default="", help="PNG with the two panels (needs matplotlib)")
    ap.add_argument("--output", default="", help="JSON with the scores of x and the changed share")
    args = ap.parse_args()
    P = np.array(A + B + C)
    y = np.array(["A"] * len(A) + ["B"] * len(B) + ["C"] * len(C))
    model = RidgeClassifier(alpha=args.alpha).fit(embed(P, args.h), y)
    T = np.vstack([embed(TERMS[c], args.h)[0] for c in model.classes_])

    def scores(points, weight):
        E = embed(points, args.h)
        return model.decision_function(E) + weight * E @ T.T
    gx, gy = np.meshgrid(np.arange(2, 760, 4), np.arange(2, 440, 4))
    grid = np.c_[gx.ravel(), gy.ravel()]
    before = scores(grid, 0).argmax(1).reshape(gx.shape)
    after = scores(grid, args.weight).argmax(1).reshape(gx.shape)
    s0, s1 = scores([X0], 0)[0], scores([X0], args.weight)[0]
    res = {"classes": list(model.classes_), "x_scores_linear": s0.round(3).tolist(), "x_scores_hybrid": s1.round(3).tolist(),
           "x_label_linear": model.classes_[s0.argmax()], "x_label_hybrid": model.classes_[s1.argmax()],
           "share_of_plane_changed": float((before != after).mean())}
    print(json.dumps(res, indent=1))
    if args.output:
        json.dump(res, open(args.output, "w"), indent=1)
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        pale = ListedColormap(["#DCE7F2", "#F6E3D6", "#ECEAE3", "#B3CDE8", "#EDC3A6", "#D3CFC2"])
        dot = {"A": "#1F5F99", "B": "#C8561E", "C": "#A8A598"}
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for ax, img, title in [(axes[0], before, "linear (ridge) only"), (axes[1], after + 3 * (after != before), f"hybrid: ridge + {args.weight:g} cos(x, t_c)")]:
            ax.imshow(img, cmap=pale, vmin=0, vmax=5, extent=(0, 760, 440, 0), interpolation="nearest")
            for c, pts in (("A", A), ("B", B), ("C", C)):
                ax.scatter(*np.array(pts).T, s=40, color=dot[c])
                ax.scatter(*TERMS[c], s=120, marker="D", color=dot[c], edgecolor="#13232F")
            ax.scatter(*X0, s=120, facecolor="white", edgecolor="#13232F", linewidth=2)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
