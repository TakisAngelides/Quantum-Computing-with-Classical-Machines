using LinearAlgebra
using Test

@enum Form begin
    left
    right
end

function initialize_MPS(N::Int64, d::Int64, D::Int64)::Vector{Array{ComplexF64}}

    """
    Initializes a random MPS with open boundary conditions of type Vector{Array{ComplexF64}} where each element of the vector is 
    a site where a 3-tensor lives or in this case a 3-array. 

    The physical index is always stored last in this 3-array and the first index is the index to the left of
    the site, while the second index is the index to the right of the site. 

    If we label the first, second and third index of this 3-array
    as 1,2,3 and call the array on each site M then schematically storage is done as:

         3
         |
    1 -- M -- 2 , where 1 and 2 are the left and right bond indices that control entanglement between sites and 3 is the physical index. 

    Note 1: This function assumes both bond indices has the same dimension.

    Note 2: The first and last sites have a left and right index respectively which is set to the value 1, ie a trivial index of 0 dimension.

    Inputs: 

    N = Number of sites (Integer)
    d = Physical index dimension (Integer)
    D = Bond dimension (Integer)

    Output:

    Vector{Array} = each element of the vector is a site where a 3-tensor lives or in this case a 3-array, representing an MPS.
    """

    mps = Vector{Array{ComplexF64}}(undef, N)
    
    # Tensor at site 1

    mps[1] = rand(ComplexF64, 1, D, d) # random 3-tensor with index dimensions 1, D and d

    # Tensor at site N

    mps[N] = rand(ComplexF64, D, 1, d)

    # Tensors at site 2 to N-1

    for i in 2:N-1
        mps[i] = rand(ComplexF64, D, D, d)
    end
    
    return mps

end

function psi_0(N::Int64, d::Int64)::Array{ComplexF64}

    """
    Prepapes a state in the form |psi> = |0>|0>...|0> where the ... signifies N kets and each ket has d degrees of freedom

    Inputs: 

    N = number of qudits

    d = degrees of freedom per qudit

    Output:

    state = Array with d^N components representing the coefficients of the state |0>...|0>
    """

    state = zeros(ComplexF64, d^N)
    state[1] = 1.0+0.0im
    state = reshape(state, Tuple(d*ones(Int64, N)))

    return state

end

function contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}

    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns
    another tensor after contracting A and B

    A: first tensor
    c_A: indices of A to contract (Tuple of Int64)
    B: second tensor
    c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should
    have the same dimension as the first index of c_B, the second index from c_A
    should have the same dimension as the second index of c_B and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the
    first index in c_B and so on.

    Note 3: If we were instead to use vectors for c_A and c_B, the memory allocation 
    sky rockets and the run time is 10 times slower. Vectors require more memory than
    tuples and run time since tuples are immutable and only store a certain type each time etc.

    Example: If A is a 4-tensor, B is a 3-tensor and I want to contract the first
    index of A with the second index of B and the fourth index of A with the first
    index of B, then the input to the contraction function should be:

    contraction(A, (1, 4), B, (2, 1))

    This will result in a 3-tensor since we have 3 open indices left after the
    contraction, namely second and third indices of A and third index of B

    Code Example:
    # @time begin
    # A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
    # B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
    # c_A = (1, 2)
    # c_B = (2, 1)
    # display(contraction(A, c_A, B, c_B))
    # end
    """

    # Get the dimensions of each index in tuple form for A and B

    A_indices_dimensions = size(A) # returns tuple(dimension of index 1 of A, ...)
    B_indices_dimensions = size(B)

    # Get the uncontracted indices of A and B named u_A and u_B. The setdiff
    # returns the elements which are in the first argument and which are not
    # in the second argument.

    u_A = setdiff(1:ndims(A), c_A)
    u_B = setdiff(1:ndims(B), c_B)

    # Check that c_A and c_B agree in length and in each of their entry they
    # have the same index dimension using the macro @assert. Below we also find
    # the dimensions of each index of the uncontracted indices as well as for the
    # contracted ones.

    dimensions_c_A = A_indices_dimensions[collect(c_A)]
    dimensions_u_A = A_indices_dimensions[collect(u_A)]
    dimensions_c_B = B_indices_dimensions[collect(c_B)]
    dimensions_u_B = B_indices_dimensions[collect(u_B)]

    @assert(dimensions_c_A == dimensions_c_B, "Note 1 in the function
    contraction docstring is not satisfied: indices of tensors to be contracted
    should have the same dimensions. Input received: indices of first tensor A
    to be contracted have dimensions $(dimensions_c_A) and indices of second
    tensor B to be contracted have dimensions $(dimensions_c_B).")

    # Permute the indices of A and B so that A has all the contracted indices
    # to the right and B has all the contracted indices to the left.

    # NOTE: The order in which we give the uncontracted indices (in this case
    # they are in increasing order) affects the result of the final tensor. The
    # final tensor will have indices starting from A's indices in increasing
    # ordera and then B's indices in increasing order. In addition c_A and c_B
    # are expected to be given in such a way so that the first index of c_A is
    # to be contracted with the first index of c_B and so on. This assumption is
    # crucial for below, since we need the aforementioned specific order for
    # c_A, c_B in order for the vectorisation below to work.

    A = permutedims(A, (u_A..., c_A...)) # Splat (...) unpacks a tuple in the argument of a function
    B = permutedims(B, (c_B..., u_B...))

    # Reshape tensors A and B so that for A the u_A are merged into 1 index and
    # the c_A are merged into a second index, making A essentially a matrix.
    # The same goes with B, so that A*B will be a vectorised implementation of
    # a contraction. Remember that c_A will form the columns of A and c_B will
    # form the rows of B and since in A*B we are looping over the columns of A
    # with the rows of B it is seen from this fact why the vectorisation works.

    # To see the index dimension of the merged u_A for example you have to think
    # how many different combinations I can have of the individual indices in
    # u_A. For example if u_A = (2, 4) this means the uncontracted indices of A
    # are its second and fourth index. Let us name them alpha and beta
    # respectively and assume that alpha ranges from 1 to 2 and beta from
    # 1 to 3. The possible combinations are 1,1 and 1,2 and 1,3 and 2,1 and 2,2
    # and 2,3 making 6 in total. In general the total dimension of u_A will be
    # the product of the dimensions of its indivual indices (in the above
    # example the individual indices are alpha and beta with dimensions 2 and
    # 3 respectively so the total dimension of the merged index for u_A will
    # be 2x3=6).

    A = reshape(A, (prod(dimensions_u_A), prod(dimensions_c_A)))
    B = reshape(B, (prod(dimensions_c_B), prod(dimensions_u_B)))

    # Perform the vectorised contraction of the indices

    C = A*B

    # Reshape the resulting tensor back to the individual indices in u_A and u_B
    # which we previously merged. This is the unmerging step.

    C = reshape(C, (dimensions_u_A..., dimensions_u_B...))

    return C

end

function mps_to_state(mps::Vector{Array{ComplexF64}}, N::Int64)::Array{ComplexF64}

    """
    If we write a quantum state as psi = psi_sigma1,sigma2...|sigma1>|sigma2>... then this function returns the tensor
    psi_sigma1,sigma2 of the psi represented by the input MPS.

    Inputs:

    mps = the mps that represents the quantum state for which we want the coefficients (Vector with elements being 3-tensors ie 3-arrays)

    N = number of lattice sites (Integer)

    Outputs:

    result = coefficients of quantum state namely the psi_sigma1,sigma2,... coefficients (Array of complex floats 64)

    """

    result = contraction(mps[1], (2,), mps[2], (1,))
    for i in 2:N-1
        result = contraction(result, (i+1,), mps[i+1], (1,))
    end

    result = contraction(ones(ComplexF64, 1), (1,), result, (1,))
    result = contraction(ones(ComplexF64, 1), (1,), result, (N,))

    return result

end

function state_to_mps(state::Array{ComplexF64}, N::Int64)::Vector{Array{ComplexF64}}

    """
    Translates the coefficients of a quantum state to mps form using SVD, see Schollwock equation (31) onwards.

    Inputs:

    state = coefficients of state to be translated to mps (Array with d^N components)

    N = number of qudits represented by state 

    Output:

    mps = N-vector with each element being an array representing the tensor on a site of the mps, this mps is the representation of the state

    """

    mps = Vector{Array{ComplexF64}}(undef, N)

    if N == 1
        
        mps[1] = reshape(state, (1, 1, size(state)[1]))

        return mps

    end

    for i in 1:N-1

        s = size(state)

        if i == 1 || length(s) <= 2
            
            left_indices = s[1]
            right_indices = s[2:end]
        
        else 

            left_indices = s[1:2]
            right_indices = s[3:end]
        
        end

        state = reshape(state, (prod(left_indices), prod(right_indices)))

        F = svd(state)

        U = F.U 
        
        S = F.S
        
        Vt = F.Vt 

        state = Diagonal(S)*Vt

        state = reshape(state, (length(S), right_indices...))

        if i == 1 && N == 2

            mps[i] = reshape(U, (1, size(U)[1], size(U)[2]))
            mps[i] = permutedims(mps[i], (1,3,2))

            mps[i+1] = reshape(state, (size(state)[1], 1, size(state)[2]))

        elseif i == 1

            mps[i] = reshape(U, (1, size(U)[1], size(U)[2]))
            mps[i] = permutedims(mps[i], (1,3,2))

        elseif i == N-1

            mps[i] = reshape(U, (left_indices..., length(S)))
            mps[i] = permutedims(mps[i], (1,3,2))

            mps[i+1] = reshape(state, (size(state)[1], 1, size(state)[2]))

        else

            mps[i] = reshape(U, (left_indices..., length(S)))
            mps[i] = permutedims(mps[i], (1,3,2))

        end

    end

    return mps

end

function gauge_site(form::Form, M_initial::Array{ComplexF64})::Tuple{Array{ComplexF64}, Array{ComplexF64}}

    """
    Gauges a site into left or right canonical form

    Note 1: See Schollwock equations (136), (137) at link: https://arxiv.org/pdf/1008.3477.pdf

    Inputs: 

    form = left or right depending on whether we want the site in left or right canonical form (of enumarative type Form)

    M_initial = 3-array to gauge representing the 3-tensor on a given site (Array)

    Output:

    If left: A, SVt # A_(a_i-1)(s_i)(sigma_i), SVt_(s_i)(a_i) and If right: US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)
    """

    # Julia is call by reference for arrays which are mutable so manipulations on M_initial in this function will reflect on the original unless we remove that reference with eg M = permutedims(M_initial, (1,2,3))

    if form == right # See Schollwock equation (137) for right canonical form (link: https://arxiv.org/pdf/1008.3477.pdf)

        D_left, D_right, d = size(M_initial) # Dimensions of indices of site represented by M_initial to be SVD decomposed
        # The next line is enough to remove the reference on M_initial so that it does not mutate the original M_initial and just uses its value, hence the gauge_site function does not mutate M_initial at all
        M = permutedims(M_initial, (1,3,2)) # Assumes initial index was left right physical and now M_(a_i-1)(sigma_i)(a_i)
        M = reshape(M, (D_left, d*D_right)) # Merging indices: Prepare as 2 index tensor to give to SVD, M_(a_i-1)(sigma_i)(a_i) -> M_(a_i-1)(sigma_i a_i)
        F = svd(M) # One can recover M by M = U*Diagonal(S)*Vt 
        U = F.U # U_(a_i-1)(s_i-1)
        S = F.S # S_(s_i-1)(s_i-1) although S here is just a vector storing the diagonal elements

        # @test length(S) == min(D_left, d*D_right) # property of SVD, note S is returned as a vector not a diagonal matrix

        # Note for complex M_initial, the following should be named Vd for V_dagger rather than Vt for V_transpose but we keep it Vt
        Vt = F.Vt # Vt_(s_i-1)(sigma_i a_i)
        Vt = reshape(Vt, (length(S), d, D_right)) # Unmerging indices: Vt_(s_i-1)(sigma_i a_i) -> Vt_(s_i-1)(sigma_i)(a_i)
        B = permutedims(Vt, (1,3,2)) # Vt_(s_i-1)(sigma_i)(a_i) -> B_(s_i-1)(a_i)(sigma_i)
        US = U*Diagonal(S) # US_(a_i-1)(s_i-1)

        # @test isapprox(contraction(B, (2,3), conj!(deepcopy(B)), (2,3)), I) # right canonical form property

        return US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)

    else # See Schollwock equation (136) for left canonical form

        D_left, D_right, d = size(M_initial) 
        M = permutedims(M_initial, (3, 1, 2)) # M_(a_i-1)(a_i)(sigma_i) -> M_(sigma_i)(a_i-1)(a_i)
        M = reshape(M, (d*D_left, D_right)) # M_(sigma_i)(a_i-1)(a_i) -> M_(sigma_i a_i-1)(a_i)
        F = svd(M)
        U = F.U # U_(sigma_i a_i-1)(s_i)
        S = F.S # S_(s_i)(s_i) although stored as vector here

        # @test length(S) == min(d*D_left, D_right) # property of SVD, note S is returned as a vector not a diagonal matrix

        Vt = F.Vt # Vt_(s_i)(a_i)
        U = reshape(U, (d, D_left, length(S))) # U_(sigma_i)(a_i-1)(s_i)
        A = permutedims(U, (2, 3, 1)) # A_(a_i-1)(s_i)(sigma_i)
        SVt = Diagonal(S)*Vt # SVt_(s_i)(a_i)

        # @test isapprox(contraction(conj!(deepcopy(A)), (1,3), A, (1,3)), I) # left canonical form property

        return A, SVt # A_(a_i-1)(s_i)(sigma_i), SVt_(s_i)(a_i)

    end

end

function gauge_mps!(form::Form, mps::Vector{Array{ComplexF64}}, normalized::Bool, N::Int64)

    """
    This function calls the function gauge_site for all sites on a lattice putting the MPS in left or right canonical form

    Inputs: 

    form = left or right depending on whether we want left or right canonical form (of enumarative type Form)

    mps = Vector{Array} representing the MPS

    normalized = true or false depending on whether we want the mutated mps to be normalized or not (Boolean)

    N = Number of physical sites on the lattice (Integer)

    Output:

    This function does not return anything. As suggested by the exclamation mark which is conventionally placed in its name (when
    the given function mutates the input), it mutates the input mps.
    """

    # In Julia, it's a convention to append ! to names of functions that modify their arguments.

    if form == right

        M_tilde = mps[N] # This will not work if the MPS site does not have 3 legs

        for i in N:-1:2 # We start from the right most site and move to the left

            US, mps[i] = gauge_site(right, M_tilde) # US will be multiplied to the M on the left
            M_tilde = contraction(mps[i-1], (2,), US, (1,)) # M_tilde_(a_i-2)(sigma_i-1)(s_i-1)
            M_tilde = permutedims(M_tilde, (1,3,2)) # Put the physical index to the right most place M_tilde_(a_i-2)(sigma_i-1)(s_i-1) -> M_tilde_(a_i-2)(s_i-1)(sigma_i-1)
            if i == 2
                if normalized # If we require the state to be normalized then we gauge even the first site to be a B tensor so that the whole contraction <psi|psi> collapses to the identity
                    _, mps[1] = gauge_site(right, M_tilde) # The placeholder _ for the value of US tells us that we are discarding that number and so the state is normalized just like we would divide psi by sqrt(a) when <psi|psi> = a
                else
                    mps[1] = M_tilde # Here we don't enforce a normalization so we dont have to gauge the first site we will just need to contract it with its complex conjugate to get the value for <psi|psi>
                end
            end
        end
    end

    if form == left 

        M_tilde = mps[1]

        for i in 1:(N-1)

            mps[i], SVt = gauge_site(left, M_tilde)
            M_tilde = contraction(SVt, (2,), mps[i+1], (1,)) # M_tilde_(s_i-1)(a_i)(sigma_i) so no permutedims needed here
            if i == (N-1)
                if normalized
                    mps[N], _ = gauge_site(left, M_tilde)
                else
                    mps[N] = M_tilde
                end
            end
        end
    end
end

function get_hadamard_mpo(N::Int64, site::Int64)::Vector{Array{ComplexF64}}

    """
    Return the mpo of the Hadamard operator H = 1/sqrt(2)[1 1; 1 -1] acting on the site specified by the site input.

    Inputs:

    N = number of qubits of circuit

    site = index of qubit on which to act the Hadamard gate

    Output:

    mpo = N-vector with each element being a 4-array representing the tensor on each site of the mpo representing the Hadamard operator acting on a given site
    """

    @assert(site >= 1 && site <= N, "The site index on which the Hadamard gate will act should be between 1 and N = $(N). The input
    given as site index was $(site).")

    zero = [0.0 0.0; 0.0 0.0]
    I = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    H = (1/sqrt(2))*(X + Z)

    mpo = Vector{Array{ComplexF64}}(undef, N)

    t1 = ones(1, 2, 2, 2)
    t1[1,1,:,:] = H
    
    ti = ones(2, 2, 2, 2)
    ti[1,1,:,:] = I
    ti[1,2,:,:] = zero
    ti[2,1,:,:] = H
    ti[2,2,:,:] = I
    
    tN = ones(2, 1, 2, 2)
    tN[1,1,:,:] = I
    tN[2,1,:,:] = H

    t1_not = zeros(1, 2, 2, 2)
    t1_not[1,2,:,:] = I

    ti_not = zeros(2, 2, 2, 2)
    ti_not[1,1,:,:] = I
    ti_not[2,2,:,:] = I

    tN_not = zeros(2,1,2,2)
    tN_not[1,1,:,:] = I

    if site == 1
    
        mpo[1] = t1

    else

        mpo[1] = t1_not
    
    end

    if site == N

        mpo[N] = tN
    
    else

        mpo[N] = tN_not

    end

    for i in 2:N-1

        if i == site
        
            mpo[i] = ti

        else

            mpo[i] = ti_not

        end

    end

    return mpo

end

function act_mpo_on_mps(mpo::Vector{Array{ComplexF64}}, mps::Vector{Array{ComplexF64}})::Vector{Array{ComplexF64}}

    """
    Act with an mpo on an mps to produce a new mps with increased bond dimension.

    Inputs:

    mpo = the mpo to act on the mps (Vector of Arrays)

    mps = the mps that the mpo will act on (Vector of Arrays)

    Output:

    result = the new mps with increased bond dimension resulting from acting with the mpo input on the mps input
    """
    
    N = length(mps)

    result = Vector{Array{ComplexF64}}(undef, N)

    for i in 1:N
    
        tmp = contraction(mpo[i], (4,), mps[i], (3,)) # Does contraction of sigma'_i: W_(b_i-1 b_i sigma_i sigma'_i) M_(a_i-1 a_i sigma'_i) = T_(b_i-1 b_i sigma_i a_i-1 a_i)
        tmp = permutedims(tmp, (4, 1, 5, 2, 3)) # T_(a_i-1 b_i-1 a_i b_i sigma_i)
        idx_dims = size(tmp) # returns a tuple of the dimensions of the indices of the tensor T_(a_i-1 b_i-1 a_i b_i sigma_i)
        result[i] = reshape(tmp, (idx_dims[1]*idx_dims[2], idx_dims[3]*idx_dims[4], idx_dims[5])) # merges the bond indices of i-1 together and the bond indices of i together by reshaping the tensor into having indices of higher dimension 

    end

    return result

end

# ---------------------------------------------------------------------------------------------------------------------

# Prepares a state |00>, acts with a Hadamard gate on the first qubit to take it to the state 0.7|00> + 0.7|10> which is displayed

N = 2
d = 2
Hadamard_mpo = get_hadamard_mpo(N, 1)
state = psi_0(N, d)
mps = state_to_mps(state, N)
mps = act_mpo_on_mps(Hadamard_mpo, mps)
state = mps_to_state(mps, N)
display(state)

# ---------------------------------------------------------------------------------------------------------------------