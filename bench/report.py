# -*- coding: UTF-8 -*-
import numpy as np
import os


def format_data(arr):
    arr = np.array(arr)
    res = []
    for j in range(4):
        res_col = []
        for i in range(2):
            if arr[i, j] == np.min(arr[:, j]):
                res_col.append("\\textbf{" + "%.2e" % arr[i, j] + "}")
            else:
                res_col.append("%.2e" % arr[i, j])
        # print(res_col)
        res.append(res_col)
    # negative counts
    """res_col = []    
    for i in range(3):
        res_col.append(str(int(arr[i, 4])))
    res.append(res_col)"""
    res.append([str(int(arr[i, 4])) for i in range(2)])
    return np.array(res).transpose()


def generate_tex_table(data, names):
    with open("table.tex", "w") as myfile:
        # header
        myfile.write(
            "\\begin{table}[h!] \n \label{table:vs} \n \\begin{scriptsize} \n \\begin{tabular}{l|c|*{5}{c}} \n"
        )
        myfile.write(
            "Demographic model & method & exec time (s) & KL div & mean($\\varepsilon_r$) & $\Delta$ LL & $<0$ entries\\\ \n \hline \n"
        )
        for i in range(len(names)):
            dt = np.array(data[i])

            myfile.write(
                "& $\dadi$ grid 1 & "
                + "%.2e" % dt[0, 0]
                + " & "
                + "%.2e" % dt[0, 1]
                + " & "
                + "%.2e" % dt[0, 2]
                + " & "
                + "%.2e" % dt[0, 3]
                + " & "
                + str(dt[0, 4])
                + " \\\ \n "
            )
            myfile.write(
                names[i]
                + " & $\dadi$ extrapolation & "
                + "%.2e" % dt[1, 0]
                + " & "
                + "%.2e" % dt[1, 1]
                + " & "
                + "%.2e" % dt[1, 2]
                + " & "
                + "%.2e" % dt[1, 3]
                + " & "
                + str(dt[1, 4])
                + " \\\ \n "
            )
            myfile.write(
                " & \\textit{Moments} & "
                + "%.2e" % dt[2, 0]
                + " & "
                + "%.2e" % dt[2, 1]
                + " & "
                + "%.2e" % dt[2, 2]
                + " & "
                + "%.2e" % dt[2, 3]
                + " & "
                + str(dt[2, 4])
                + " \\\ \n \hline \n"
            )
        # end of the table
        myfile.write(
            "\end{tabular} \n \end{scriptsize}"
            + "\caption{Performance comparisons between $\dadi$ and \\textit{Moments} on several scenarios. "
            + "For $\dadi$ simulations we use $\gamma \\times ns$ number of grid points for each dimension, "
            + '$ns$ being the sample size per population (30 here).  "$\dadi$ grid 1" corresponds to '
            + "simulations with $\gamma=1.5$ for neutral cases or $\gamma=2$ for cases with selection "
            + 'whereas "$\dadi$ grid 2" are performed with a finer grid $\gamma=5$} \n'
            + "\end{table}"
        )


def generate_formated_table(data, names):
    with open("table.tex", "w") as myfile:
        # header
        myfile.write(
            "\\begin{table}[h!] \n \label{table:vs} \n \\begin{scriptsize} \n \\begin{tabular}{l|c|*{5}{c}} \n"
        )
        myfile.write(
            "Demographic model & method & exec time (s) & KL div & mean($\\varepsilon_r$) & $\Delta$ LL & $<0$ entries\\\ \n \hline \n"
        )
        for i in range(len(names)):
            dt = format_data(data[i])
            """
            myfile.write('& $\dadi$ grid 1 & '+dt[0, 0]+' & '+dt[0, 1]
                         +' & '+dt[0, 2]+' & '+dt[0, 3]+' & '+dt[0, 4]+' \\\ \n ')"""
            myfile.write(
                names[i]
                + " & $\dadi$ extrapolation & "
                + dt[0, 0]
                + " & "
                + dt[0, 1]
                + " & "
                + dt[0, 2]
                + " & "
                + dt[0, 3]
                + " & "
                + dt[0, 4]
                + " \\\ \n "
            )
            myfile.write(
                " & \\textit{Moments} & "
                + dt[1, 0]
                + " & "
                + dt[1, 1]
                + " & "
                + dt[1, 2]
                + " & "
                + dt[1, 3]
                + " & "
                + dt[1, 4]
                + " \\\ \n \hline \n"
            )
        # end of the table
        myfile.write(
            "\end{tabular} \n \end{scriptsize}"
            + "\caption{Performance comparisons between $\dadi$ and \\textit{Moments} on several scenarios. "
            + "For $\dadi$ simulations we use $\gamma \\times ns$ number of grid points for each dimension, "
            + '$ns$ being the sample size per population (30 here).  "$\dadi$ grid 1" corresponds to '
            + "simulations with $\gamma=1.5$ for neutral cases or $\gamma=2$ for cases with selection "
            + 'whereas "$\dadi$ grid 2" are performed with a finer grid $\gamma=5$} \n'
            + "\end{table}"
        )
