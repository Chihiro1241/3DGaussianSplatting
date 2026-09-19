"""Render scene_memory.csv as a LaTeX table with \\multirow dataset groups."""
import csv
from itertools import groupby
from pathlib import Path

CSV = Path("output/3DGS/benchmark_report/report/scene_memory.csv")
OUT = Path("output/3DGS/benchmark_report/report/scene_memory_table.tex")

DATASET_ORDER = ["Mip-NeRF360", "Tanks&Temples", "Deep Blending", "Synthetic NeRF"]
LATEX_DATASET = {"Tanks&Temples": r"Tanks\&Temples"}

HEADER = r"""% \usepackage{multirow} が必要（Dataset 列の \multirow によるグループ化に使用）
% \usepackage{float} も必要（table 環境の [H] 指定に使用）
\begin{table}[H]
    \centering
    \caption{Scene別のGaussian数および学習時メモリ使用量（30,000反復時点）。
             Peak allocated はPyTorchが実際に使用したVRAMの最大値、
             Peak reserved はCUDAアロケータが確保した最大値、
             Peak process はOSから観測したプロセス全体のVRAM最大値を表す。}
    \label{tab:scene_memory}
    \begin{tabular}{llrrrr}
        \hline
        Dataset & Scene & Gaussian count
        & Peak allocated (MiB) & Peak reserved (MiB) & Peak process (MiB) \\
        \hline
"""

FOOTER = r"""        \hline
    \end{tabular}
\end{table}
"""


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    rows.sort(key=lambda r: DATASET_ORDER.index(r["dataset"]))

    lines = []
    for index, (dataset, group) in enumerate(groupby(rows, key=lambda r: r["dataset"])):
        group = list(group)
        if index:
            lines.append(r"        \hline")
        label = LATEX_DATASET.get(dataset, dataset)
        lines.append(f"        % {label}（{len(group)}シーン）")
        for offset, row in enumerate(group):
            first = rf"\multirow{{{len(group)}}}{{*}}{{{label}}}" if offset == 0 else ""
            cells = [
                first,
                row["scene"],
                f"{int(row['gaussian_count']):,}",
                f"{float(row['peak_allocated_MiB']):,.1f}",
                f"{float(row['peak_reserved_MiB']):,.1f}",
                f"{float(row['peak_process_vram_MiB']):,.1f}",
            ]
            lines.append("        " + " & ".join(cells) + r" \\")

    OUT.write_text(HEADER + "\n".join(lines) + "\n" + FOOTER)
    print(f"wrote {OUT} ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
