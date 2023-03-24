import numpy as np


def D(counts):
    c1, c2, c3, c4 = counts
    n = sum(counts)
    numer = c1 * c4 - c2 * c3
    return 1.0 * numer / n / (n - 1)


def DD(counts, pop_nums):
    if len(pop_nums) != 2:
        raise ValueError("problem with DD")
    pop1, pop2 = pop_nums
    if pop1 == pop2:
        # compute D^2 for pop in pop_nums
        cs = counts[pop1]
        c1, c2, c3, c4 = cs
        n = sum(cs)
        numer = (
            # (c1 * c4 - c2 * c3) ** 2 - c2 * c3 * (c2 + c3 - 1) - c1 * c4 * (c1 + c4 - 1)
            c1 * (c1 - 1) * c4 * (c4 - 1)
            + c2 * (c2 - 1) * c3 * (c3 - 1)
            - 2 * c1 * c2 * c3 * c4
        )
        return 1.0 * numer / n / (n - 1) / (n - 2) / (n - 3)
    else:
        cs1, cs2 = counts[pop1], counts[pop2]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (c12 * c13 - c11 * c14) * (c22 * c23 - c21 * c24)
        return 1.0 * numer / n1 / (n1 - 1) / n2 / (n2 - 1)


def Dz(counts, pop_nums):
    if len(pop_nums) != 3:
        raise ValueError("error in Dz")
    pop1, pop2, pop3 = pop_nums
    if pop1 == pop2 == pop3:
        cs = counts[pop1]
        c1, c2, c3, c4 = cs
        n = sum(cs)
        numer = (
            (c1 * c4 - c2 * c3) * (c3 + c4 - c1 - c2) * (c2 + c4 - c1 - c3)
            + (c1 * c4 - c2 * c3) * (c2 + c3 - c1 - c4)
            + 2 * (c2 * c3 + c1 * c4)
        )
        return 1.0 * numer / n / (n - 1) / (n - 2) / (n - 3)
    elif pop1 == pop2:  # Dz(i,i,j)
        cs1, cs2 = counts[pop1], counts[pop3]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (
            (-c11 - c12 + c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 + c22 - c23 + c24)
        )
        return 1.0 * numer / n2 / n1 / (n1 - 1) / (n1 - 2)
    elif pop1 == pop3:  # Dz(i,j,i)
        cs1, cs2 = counts[pop1], counts[pop2]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (
            (-c11 + c12 - c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 - c22 + c23 + c24)
        )
        return 1.0 * numer / n2 / n1 / (n1 - 1) / (n1 - 2)
    elif pop2 == pop3:  # Dz(i,j,j)
        cs1, cs2 = counts[pop1], counts[pop2]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (-(c12 * c13) + c11 * c14) * (-c21 + c22 + c23 - c24) + (
            -(c12 * c13) + c11 * c14
        ) * (-c21 + c22 - c23 + c24) * (-c21 - c22 + c23 + c24)
        return 1.0 * numer / n1 / (n1 - 1) / n2 / (n2 - 1)
    else:  # Dz(i,j,k)
        cs1 = counts[pop1]
        cs2 = counts[pop2]
        cs3 = counts[pop3]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        c31, c32, c33, c34 = cs3
        n1 = sum(cs1)
        n2 = sum(cs2)
        n3 = sum(cs3)
        numer = -(
            (
                (c12 * c13 - c11 * c14)
                * (c21 + c22 - c23 - c24)
                * (c31 - c32 + c33 - c34)
            )
        )
        return 1.0 * numer / n1 / (n1 - 1) / n2 / n3


def pi2(counts, pop_nums):
    if len(pop_nums) != 4:
        raise ValueError("mistake in pi2")
    pop1, pop2, pop3, pop4 = pop_nums
    if pop1 == pop2 == pop3 == pop4:
        cs = counts[pop1]
        c1, c2, c3, c4 = cs
        n = sum(cs)
        numer = (
            (c1 + c2) * (c1 + c3) * (c2 + c4) * (c3 + c4)
            - c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
            - c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4)
        )
        return 1.0 * numer / n / (n - 1) / (n - 2) / (n - 3)
    elif (
        (pop1 == pop2 == pop3)
        or (pop1 == pop2 == pop4)
        or (pop1 == pop3 == pop4)
        or (pop2 == pop3 == pop4)
    ):  # pi2(i,i;i,j) or pi2(i,i;j,i) or pi2(i,j;i,i) or pi2(j,i;i,i)
        if pop1 == pop2 == pop3:
            cs1, cs2 = counts[pop1], counts[pop4]
        elif pop1 == pop2 == pop4:
            cs1, cs2 = counts[pop1], counts[pop3]
        elif pop1 == pop3 == pop4:
            cs1, cs2 = counts[pop1], counts[pop2]
        elif pop2 == pop3 == pop4:
            cs1, cs2 = counts[pop2], counts[pop1]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (
            -((c11 + c12) * c14 * (c21 + c23))
            - (c12 * (c13 + c14) * (c21 + c23))
            + ((c11 + c12) * (c12 + c14) * (c13 + c14) * (c21 + c23))
            + ((c11 + c12) * (c13 + c14) * (-2 * c22 - 2 * c24))
            + ((c11 + c12) * c14 * (c22 + c24))
            + (c12 * (c13 + c14) * (c22 + c24))
            + ((c11 + c12) * (c11 + c13) * (c13 + c14) * (c22 + c24))
        ) / 2.0
        return 1.0 * numer / n2 / n1 / (n1 - 1) / (n1 - 2)

    elif pop1 == pop2 and pop3 == pop4:  # pi2(i,i;j,j)
        cs1, cs2 = counts[pop1], counts[pop3]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (c11 + c12) * (c13 + c14) * (c21 + c23) * (c22 + c24)
        return 1.0 * numer / n1 / (n1 - 1) / n2 / (n2 - 1)

    elif (pop1 == pop3 and pop2 == pop4) or (  # pi2(i,j;i,j) or
        pop1 == pop4 and pop2 == pop3
    ):  # pi2(i,j;j,i)
        cs1, cs2 = counts[pop1], counts[pop2]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        n1 = sum(cs1)
        n2 = sum(cs2)
        numer = (
            ((c12 + c14) * (c13 + c14) * (c21 + c22) * (c21 + c23)) / 4.0
            + ((c11 + c13) * (c13 + c14) * (c21 + c22) * (c22 + c24)) / 4.0
            + ((c11 + c12) * (c12 + c14) * (c21 + c23) * (c23 + c24)) / 4.0
            + ((c11 + c12) * (c11 + c13) * (c22 + c24) * (c23 + c24)) / 4.0
            + (
                -(c12 * c13 * c21)
                + c14 * c21
                - c12 * c14 * c21
                - c13 * c14 * c21
                - c14 ** 2 * c21
                - c14 * c21 ** 2
                + c13 * c22
                - c11 * c13 * c22
                - c13 ** 2 * c22
                - c11 * c14 * c22
                - c13 * c14 * c22
                - c13 * c21 * c22
                - c14 * c21 * c22
                - c13 * c22 ** 2
                + c12 * c23
                - c11 * c12 * c23
                - c12 ** 2 * c23
                - c11 * c14 * c23
                - c12 * c14 * c23
                - c12 * c21 * c23
                - c14 * c21 * c23
                - c11 * c22 * c23
                - c14 * c22 * c23
                - c12 * c23 ** 2
                + c11 * c24
                - c11 ** 2 * c24
                - c11 * c12 * c24
                - c11 * c13 * c24
                - c12 * c13 * c24
                - c12 * c21 * c24
                - c13 * c21 * c24
                - c11 * c22 * c24
                - c13 * c22 * c24
                - c11 * c23 * c24
                - c12 * c23 * c24
                - c11 * c24 ** 2
            )
            / 4.0
        )
        return 1.0 * numer / n1 / (n1 - 1) / n2 / (n2 - 1)

    elif pop1 == pop2:  # pi2(i,i;j,k)
        cs1 = counts[pop1]
        cs2 = counts[pop3]
        cs3 = counts[pop4]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        c31, c32, c33, c34 = cs3
        n1 = sum(cs1)
        n2 = sum(cs2)
        n3 = sum(cs3)
        numer = (
            (c11 + c12)
            * (c13 + c14)
            * (c22 * (c31 + c33) + c24 * (c31 + c33) + (c21 + c23) * (c32 + c34))
        ) / 2.0
        return 1.0 * numer / n1 / (n1 - 1) / n2 / n3

    elif pop3 == pop4:  # pi2(i,j;k,k)
        cs1 = counts[pop3]
        cs2 = counts[pop1]
        cs3 = counts[pop2]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        c31, c32, c33, c34 = cs3
        n1 = sum(cs1)
        n2 = sum(cs2)
        n3 = sum(cs3)
        numer = (
            (c11 + c13)
            * (c12 + c14)
            * (c23 * (c31 + c32) + c24 * (c31 + c32) + (c21 + c22) * (c33 + c34))
        ) / 2.0
        return 1.0 * numer / n1 / (n1 - 1) / n2 / n3

    elif (
        (pop1 == pop3) or (pop1 == pop4) or (pop2 == pop3) or (pop2 == pop4)
    ):  # pi2(i,j;i,k) or pi2(i,j;k,i) or pi2(i,j;j,k) or pi2(i,j;k,j)
        if pop1 == pop3:  # pi2(i,j;i,k)
            cs1 = counts[pop1]
            cs2 = counts[pop2]
            cs3 = counts[pop4]
        elif pop1 == pop4:  # pi2(i,j;k,i)
            cs1 = counts[pop1]
            cs2 = counts[pop2]
            cs3 = counts[pop3]
        elif pop2 == pop3:  # pi2(i,j;j,k)
            cs1 = counts[pop2]
            cs2 = counts[pop1]
            cs3 = counts[pop4]
        elif pop2 == pop4:  # pi2(i,j;k,j)
            cs1 = counts[pop2]
            cs2 = counts[pop1]
            cs3 = counts[pop3]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        c31, c32, c33, c34 = cs3
        n1 = sum(cs1)
        n2 = sum(cs2)
        n3 = sum(cs3)
        numer = (
            c14 ** 2 * (c21 + c22) * (c31 + c33)
            + c12 ** 2 * (c23 + c24) * (c31 + c33)
            + (-1 + c11 + c13) * (c13 * (c21 + c22) + c11 * (c23 + c24)) * (c32 + c34)
            + c14
            * (
                c11 * (c23 + c24) * (c31 + c33)
                + c21
                * (
                    (-1 + c13) * c31
                    + c13 * c32
                    - c33
                    + c13 * c33
                    + c13 * c34
                    + c11 * (c32 + c34)
                )
                + c22
                * (
                    (-1 + c13) * c31
                    + c13 * c32
                    - c33
                    + c13 * c33
                    + c13 * c34
                    + c11 * (c32 + c34)
                )
            )
            + c12
            * (
                c14 * (c21 + c22 + c23 + c24) * (c31 + c33)
                + c13
                * (c21 * (c31 + c33) + c22 * (c31 + c33) + (c23 + c24) * (c32 + c34))
                + (c23 + c24) * ((-1 + c11) * c31 - c33 + c11 * (c32 + c33 + c34))
            )
        ) / 4.0
        return 1.0 * numer / n1 / (n1 - 1) / n2 / n3

    else:  # pi2(i,j,k,l)
        cs1, cs2, cs3, cs4 = counts[pop1], counts[pop2], counts[pop3], counts[pop4]
        c11, c12, c13, c14 = cs1
        c21, c22, c23, c24 = cs2
        c31, c32, c33, c34 = cs3
        c41, c42, c43, c44 = cs4
        n1 = sum(cs1)
        n2 = sum(cs2)
        n3 = sum(cs3)
        n4 = sum(cs4)
        numer = (
            ((c13 + c14) * (c21 + c22) * (c32 + c34) * (c41 + c43)) / 4.0
            + ((c11 + c12) * (c23 + c24) * (c32 + c34) * (c41 + c43)) / 4.0
            + ((c13 + c14) * (c21 + c22) * (c31 + c33) * (c42 + c44)) / 4.0
            + ((c11 + c12) * (c23 + c24) * (c31 + c33) * (c42 + c44)) / 4.0
        )
        return 1.0 * numer / n1 / n2 / n3 / n4
