import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Compute the qr iteration of a matrix to find eigenvalues and eigenvectors
'''

threshold = 1e-10
def generate_householder_matrix(first_column_prime_vector, iteration_count):
    I = np.eye(len(first_column_prime_vector))
    basis_vector = np.zeros(len(first_column_prime_vector))
    basis_vector[0] = 1

    w = 0.5*(np.linalg.norm(first_column_prime_vector)*basis_vector - first_column_prime_vector)
    
    if np.linalg.norm(w) < threshold:
        w = basis_vector*0.01

    parent_I = np.eye(np.size(I,0)+iteration_count)
    block = I-2*np.outer(w,w) / np.linalg.norm(w) / np.linalg.norm(w)
    parent_I[iteration_count:,iteration_count:] = block
    return parent_I


def householder_transformation(householder_matrix, matrix):
    transformed = np.matmul(householder_matrix, matrix)   
    transformed = np.matmul(transformed, np.transpose(householder_matrix))
    return transformed


def update_heatmap(data):
    plt.clf()  # Clear the current figure
    sns.heatmap(data, annot=False, cmap="coolwarm", cbar=True)
    plt.draw()  # Update the figure
    plt.pause(0.1)  # Pause to allow the figure to update
    plt.show()


def hessenberg(matrix, iteration_count, accumulated_householder_matrices):
    update_heatmap(matrix)
    if iteration_count == len(matrix)-1:
        return matrix, accumulated_householder_matrices

    column = matrix[:, iteration_count-1]
    # print(column, "column")
    column_prime_vector = column[iteration_count:]
    # print(column_prime_vector, "column_prime_vector")
    
    householder_matrix = generate_householder_matrix(column_prime_vector, iteration_count)
    accumulated_householder_matrices = np.matmul(accumulated_householder_matrices, householder_matrix)

    transformed_this_itr = householder_transformation(householder_matrix, matrix)

    transformed_next_itr, accumulated_householder_matrices = hessenberg(transformed_this_itr, iteration_count+1,accumulated_householder_matrices)

    return transformed_next_itr,accumulated_householder_matrices


def gram_schmidt(matrix):
    orthogonal_vectors = []
    for idx, col_vector in enumerate(matrix.T):
        if idx == 0:
            new_orthogonal_vector = col_vector / np.linalg.norm(col_vector)
            orthogonal_vectors.append(new_orthogonal_vector)
        else:
            new_orthogonal_vector = col_vector
            for previous_orthogonal_vector in orthogonal_vectors:
                projected_vector = np.dot(col_vector, previous_orthogonal_vector)*previous_orthogonal_vector
                new_orthogonal_vector = new_orthogonal_vector - projected_vector 

            new_orthogonal_vector = new_orthogonal_vector / np.linalg.norm(new_orthogonal_vector)
            orthogonal_vectors.append(new_orthogonal_vector)

    for i in range(len(orthogonal_vectors)):
        vector = orthogonal_vectors[i]
        for k in range(len(orthogonal_vectors)):
            if i == k:
                continue
            other_vector = orthogonal_vectors[k]

            assert(np.dot(vector, other_vector)<threshold)

    return orthogonal_vectors

def qr_decompose(matrix):
    orthogonal_vectors = gram_schmidt(matrix)
    vector_len = len(matrix)
    Q = np.column_stack(tuple(orthogonal_vectors))
    R = np.empty((vector_len, 0))
    for k in range(len(orthogonal_vectors)):
        R_vector = np.zeros(vector_len)
        vector = matrix[:,k]
        for i in range(k+1):
            R_vector[i] = np.dot(vector, orthogonal_vectors[i])
        R = np.column_stack((R, R_vector))

    return Q,R


def compute_wilkinson_shift(sub_matrix, trailing_value):
    a = sub_matrix[0,0]
    b = sub_matrix[0,1]
    c = sub_matrix[1,0]
    d = sub_matrix[1,1]
    first_eigenvalue = (a + c + np.sqrt((a+c)**2-4*(a*c-b*d)))/ 2.0
    second_eigenvalue = (a + c - np.sqrt((a+c)**2-4*(a*c-b*d)))/ 2.0
    # return 0.001
    if not np.iscomplex(first_eigenvalue) and not np.iscomplex(second_eigenvalue):
        if np.abs(first_eigenvalue-trailing_value) < np.abs(second_eigenvalue-trailing_value):
            return first_eigenvalue
        else:
            return second_eigenvalue
    else:
        return np.real(first_eigenvalue)

def qr_algo(A_n, with_shift, accumulated_Q):
    if with_shift:
        sub_matrix = A_n[-2:, -2:]
        shift = compute_wilkinson_shift(sub_matrix, sub_matrix[1,1])
        shifted_matrix = A_n - shift*np.eye(len(A_n))
        Q,R = qr_decompose(shifted_matrix)
        
        A_n_plus_one = np.matmul(R,Q) + shift*np.eye(len(A_n))
        accumulated_Q = np.matmul(accumulated_Q, Q)

    else:
        Q,R = qr_decompose(A_n)
        A_n_plus_one = np.matmul(R,Q) 
        accumulated_Q = np.matmul(accumulated_Q, Q)
    
    return A_n_plus_one, accumulated_Q

def extract_eigenvalues(A_n_plus_one):
    return np.diag(A_n_plus_one)

rows, cols = 20,20
matrix = np.random.rand(rows,cols)
matrix = (matrix + matrix.T) / 2
print("First, we generate a symmetric matrix guaranteed to have n eigenvalues")
print(matrix)
print()

plt.ion()
hessenberg_matrix, accumulated_householder_matrices = hessenberg(matrix,1,np.eye(len(matrix)))
update_heatmap(hessenberg_matrix)

# for displaying hessenberg matrix
print("Reduce the symmetric matrix into hessenberg form")
h_matrix = hessenberg_matrix
h_matrix[np.abs(h_matrix) < threshold] = 0
print(h_matrix)
print()

with_shift = False
A_n_plus_one, accumulated_Q = qr_algo(hessenberg_matrix,with_shift, np.eye(len(hessenberg_matrix)))
update_heatmap(A_n_plus_one)
for i in range(100):
    A_n_plus_one,accumulated_Q = qr_algo(A_n_plus_one,with_shift,accumulated_Q)
    update_heatmap(A_n_plus_one)
    

print("Converged matrix is")
converged_matrix = A_n_plus_one
converged_matrix[np.abs(converged_matrix) < threshold] = 0
print(converged_matrix)
print()


print("Our generated eigenvalues")
computed_eigenvalues = extract_eigenvalues(A_n_plus_one)
print(computed_eigenvalues)
print()

print("Our generated eigenvectors vvvv")
for i, eigenvector in enumerate(np.matmul(accumulated_householder_matrices, accumulated_Q).T):
    print("eigenvector X=",eigenvector)
    print("eigenvalue=", computed_eigenvalues[i])
    print("AX=",np.matmul(matrix,eigenvector))
    print("λX=",eigenvector*computed_eigenvalues[i])
    print("------------------------------------------------")

print()

print("Generated eigenvalues with numpy library")
eigenvalues = np.linalg.eigvals(matrix)
print()
print("Generated eigenvectors with numpy library")
eigenvectors = np.linalg.eig(matrix).eigenvectors
print(eigenvalues)
print(eigenvectors)

