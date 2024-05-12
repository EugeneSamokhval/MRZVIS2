import random
import math

m, p, q, n = 0, 0, 0, 0


timeOfSumm = 1
timeOfDifference = 1
timeOfMultiplying = 1
timeOfCom = 1

callsOfSumm = 0
callsOfDifference = 0
callsOfMultiplying = 0
callsOfCom = 0

T1 = 0
Tn = 0
Ky = 0
Eff = 0
Diff = 0
Lavg = 0


def operation_and(matrix_1, matrix_2, k, i, j):
    result = matrix_1[i][k] + matrix_2[k][j] - 1
    if result < 0:
        result = 0
    global callsOfCom, callsOfSumm, callsOfDifference
    callsOfCom += 1
    callsOfSumm += 1
    callsOfDifference += 1
    return result


def more(matrix_1, i, k) -> float:
    global callsOfCom
    callsOfCom += 1
    if matrix_1[i][k] >= 0:
        return 0.
    else:
        return matrix_1[i][k]


def more_inverse(matrix_1, i, k) -> float:
    global callsOfCom
    callsOfCom += 1
    if matrix_1[k][i] >= 0:
        return 0.
    else:
        return matrix_1[k][i]


def function_F_of_i_j_k(matrix_F, matrix_1, matrix_2, matrix_3, i, j, k):
    global callsOfMultiplying, callsOfDifference, callsOfSumm
    callsOfMultiplying += 7
    callsOfDifference += 3
    callsOfSumm += 2
    matrix_F[i][j][k] = more(matrix_2, k, j) * (2. * matrix_3[k] - 1.) * matrix_3[k] + more_inverse(
            matrix_1, k, i) * (1. + (4. * more(matrix_2, k, j) - 2.) * matrix_3[k]) * (1. - matrix_3[k])
    return matrix_F


def functon_D_of_i_j_k(matrix_D, matrix_1, matrix_2, i, j, k):
    matrix_D: list
    matrix_D[i][j][k] = operation_and(matrix_1, matrix_2, k, i, j)
    return matrix_D


def D_multiplication(matrix_D, i, j):
    result = 0
    global callsOfDifference, callsOfMultiplying, callsOfDifference
    callsOfDifference += m + 1
    callsOfMultiplying += m - 1
    callsOfDifference += 1
    for k in range(m):
        result *= 1 - matrix_D[i][j][k]
    return 1 - result


def F_multiplication(matrix_F, i, j):
    result = 0
    global callsOfMultiplying
    callsOfMultiplying += m - 1
    for k in range(m):
        result *= matrix_F[i][j][k]
    return result


def F_D_multiplication(matrix_F, matrix_D, i, j):
    return F_multiplication(matrix_F, i, j) * D_multiplication(matrix_D, i, j)


def C_funtion(matrix_G, matrix_F, matrix_D, i, j):
    global callsOfMultiplying, callsOfDifference, callsOfSumm
    callsOfMultiplying += 8
    callsOfDifference += 3
    callsOfSumm += 2
    return F_multiplication(matrix_F, i, j) * (3. * matrix_G[i][j] - 2.) * matrix_G[i][j] + (D_multiplication(matrix_D, i, j) +
                                                                                             (4. * F_D_multiplication(matrix_F, matrix_D, i, j) - 3. * D_multiplication(matrix_D, i, j)) * matrix_G[i][j]) * (1. - matrix_G[i][j])


def matrix_output(matrix, name):
    print("Matrix " + name + "\n"+50*"__")
    for row in matrix:
        try:
            print(*row)
        except TypeError:
            print(row)
    print(50*"__"+"\n")


def ProgramAttributesCounterResult(p, q, m, r):
    global T1, timeOfSumm, callsOfSumm, timeOfDifference, callsOfDifference, timeOfMultiplying, callsOfMultiplying, timeOfCom, callsOfCom, Tn, Ky, Eff, Lavg, Diff
    T1 = timeOfSumm * callsOfSumm + timeOfDifference * callsOfDifference + \
        timeOfMultiplying * callsOfMultiplying + timeOfCom * callsOfCom
    if (Tn > T1):
        Tn = T1
    Ky = T1 / Tn
    Eff = Ky / n
    # D
    Lavg = timeOfMultiplying * r
    # F
    Lavg += (7 * timeOfMultiplying + 3 * timeOfDifference + 2 * timeOfSumm) * r
    # C
    Lavg += (8 * timeOfMultiplying + 3 *
             timeOfDifference + 2 * timeOfSumm) * p * q
    # a_or_b
    Lavg += (timeOfCom + timeOfSumm + timeOfDifference) * (m - 1) * 2 * p * q
    # D_func
    Lavg += (timeOfMultiplying * (m - 1) +
             timeOfDifference * (m + 1)) * 3 * p * q
    # a_to_b
    Lavg += (timeOfCom)*r * 3
    Lavg = math.ceil(Lavg / r)
    Diff = Tn / Lavg

    print(
        f"T1= {T1}\nTn= {Tn}\nKy= {Ky}\ne= {Eff}\nLsum= {Tn}\nLavg= {Lavg}\nD= {Diff}")


def main():
    global Tn, m, p, q, n
    m, p, q, n = [int(entry) for entry in input(
        "Введите m p q n через пробел\n").split(' ')]
    r = p * m * q
    matrix_A = [[random.uniform(1, -1) for j in range(m)] for i in range(p)]
    matrix_B = [[random.uniform(1, -1) for j in range(q)] for i in range(m)]
    matrix_E = [random.uniform(1, -1) for j in range(m)]
    matrix_F = [[[0 for k in range(m)] for j in range(q)] for i in range(p)]
    matrix_C = [[0 for j in range(q)] for i in range(p)]
    matrix_D = [[[0 for k in range(m)] for j in range(q)] for i in range(p)]
    matrix_G = [[random.uniform(1, -1) for j in range(q)] for i in range(p)]

    matrix_output(matrix=matrix_A, name='A')
    matrix_output(matrix=matrix_B, name='B')
    matrix_output(matrix=matrix_E, name='E')
    matrix_output(matrix=matrix_G, name='G')

    for i in range(p):
        for j in range(q):
            for k in range(m):
                matrix_F = function_F_of_i_j_k(matrix_F, matrix_A,
                                               matrix_B, matrix_E, i, j, k)

    for i in range(p):
        for j in range(q):
            for k in range(m):
                matrix_D = functon_D_of_i_j_k(
                    matrix_D, matrix_A, matrix_B, i, j, k)

    reductionTime = 3 * (timeOfMultiplying + timeOfDifference + timeOfSumm)
    operationTime = 7 * timeOfMultiplying + 3 * timeOfDifference + 2 * timeOfSumm

# Update Tn
    Tn += (reductionTime + operationTime) * math.ceil(r / n)

    for i in range(p):
        for j in range(q):
            matrix_C[i][j] = C_funtion(matrix_G, matrix_F, matrix_D, i, j)

    FTime = 2 * timeOfMultiplying * (m - 1)
    DTime = 3 * (timeOfDifference * (m + 1) + timeOfMultiplying * (m - 1))
    operationTime = 8 * timeOfMultiplying + 4 * timeOfDifference + 2 * timeOfSumm
    Tn += (FTime + DTime + operationTime) * math.ceil((p * q) / n)
    matrix_output(matrix=matrix_C, name='C')

    ProgramAttributesCounterResult(p, q, m, r)


if __name__ == "__main__":
    main()
