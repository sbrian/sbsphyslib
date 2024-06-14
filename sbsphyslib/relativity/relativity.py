from sympy import Matrix, Rational, derive_by_array, diff, simplify

def calculate_christoffel(metric_matrix: Matrix, coord_symbols: list) -> list:
    """
    Parameters:
    metric_matrix : sympy.Matrix
        The metric that is the matrix. Include signs.
    coord_symbols : list of sympy.Symbol
        The coordinate symbols as simpy.symbol objects
    Return:
    list
        A four-dimensional array corresponding to the Christoffel symbol
    """

    inverse_metric_matrix = metric_matrix.inv()

    # Compute the partial derivatives of the metric tensor
    # In the partial_derivatives[i][j][k]
    # i is the variable being differentiated
    # and j,k are the index into the metric
    partial_derivatives = [derive_by_array(metric_matrix, coord) for coord in coord_symbols]

    # Initialize a tensor to store Christoffel symbols
    # in christoffel_symbols[i][j][k]
    # i is top, j is bottom left and k is bottom right
    christoffel_symbols = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]

    # Calculate each Christoffel symbol
    # i is the upper index and j and k are the two lower indices.
    # The order of j, k doesn't matter, because symmetric
    for i in range(4):
        for j in range(4):
            for k in range(4):
                sum_expr = 0
                for l in range(4):
                    sum_expr += inverse_metric_matrix[i, l] * (partial_derivatives[j][k, l] 
                        + partial_derivatives[k][j, l] - partial_derivatives[l][k, j])
                christoffel_symbols[i][j][k] = Rational(1, 2) * sum_expr
    return christoffel_symbols

def calculate_riemann_curvature_tensor(metric_matrix: Matrix, coord_symbols: list) -> list:
    """
    Parameters:
    metric_matrix : sympy.Matrix
        The metric that is the matrix. Include signs.
    coord_symbols : list of sympy.Symbol
        The coordinate symbols as simpy.symbol objects
    Return:
    list
        A four-dimensional array corresponding to the tensor
    """

    # Initialize a tensor to store the Riemann Curvature Tensor values
    # in christoffel_symbols[i][j][k][l]
    # i is top, j is bottom left and k is bottom middle and l is bottom right
    riemann_curvature_tensor = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    
    # Get the Christoffel symbols we need to populate the tensor
    christoffel_symbols = calculate_christoffel(metric_matrix, coord_symbols)

    def sum_two_christoffel_entries(i, x, y, z):
        sum = 0
        for m in range(4):
            sum = sum + christoffel_symbols[m][x][y] * christoffel_symbols[i][z][m]
        return sum

    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    riemann_curvature_tensor[i][j][k][l] = (
                        diff(christoffel_symbols[i][l][j], coord_symbols[k])
                        - diff(christoffel_symbols[i][k][j], coord_symbols[l])
                        + sum_two_christoffel_entries(i, l, j, k)
                        - sum_two_christoffel_entries(i, k, j, l))
    
    return riemann_curvature_tensor

def calculate_ricci_tensor(metric_matrix: Matrix, coord_symbols: list):
    """
    Parameters:
    metric_matrix : sympy.Matrix
        The metric that is the matrix. Include signs.
    coord_symbols : list of sympy.Symbol
        The coordinate symbols as simpy.symbol objects
    Return:
    list
        A four-dimensional array corresponding to the tensor
    """

    riemann_curvature_tensor = calculate_riemann_curvature_tensor(metric_matrix, coord_symbols)

    # Initialize a tensor to store the Ricci Tensor values
    # in christoffel_symbols[i][j][k][l]
    # i is top, j is bottom left and k is bottom middle and l is bottom right
    ricci_tensor = [[0 for _ in range(4)] for _ in range(4)]

    # Populate the entries by summing over the 2nd and 4th components of the Riemann
    # Curvature Tensor
    for i in range(4):
        for j in range(4):
            ricci_tensor[i][j] = simplify( riemann_curvature_tensor[0][i][0][j] 
            + riemann_curvature_tensor[1][i][1][j] + riemann_curvature_tensor[2][i][2][j] 
            + riemann_curvature_tensor[3][i][3][j] )

    return ricci_tensor

