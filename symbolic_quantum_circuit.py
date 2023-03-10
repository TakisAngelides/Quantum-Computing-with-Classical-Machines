from sympy.functions.elementary import trigonometric as sympy_trig
from sympy.functions.elementary import exponential as sympy_exp
import numpy as np
from sympy import *

def symbolic_circuit_evaluation(n, gates, initial_state):
    
    """
    
    Inputs:
    
    n = number of qubits
    
    gates =  a list of gates where each element of this list should specify the type of gate as a string, the parameter theta (if parametric gate) which can be a sympy symbol or a value and the index of the qubit to apply the gate
    eg  gates = [['cnot', 0, 2], ['rx', 0.5*np.pi, 1], ['x', 1]] and gates are applied in the order cnot, rx, x here, note that the list for the cnot specifies first the control qubit index and then the target

    initial_state = it is a 2**n vector (np.array) that represents the initial state on which the circuit will act
    
    Notes: 
    
    1) The qubits are ordered as eg for n = 5: |43210>
    
    2) The necessary imports are 
    
    3) The available gates are parametric rotations rx, ry, rz, the parametric phase gate p, then x, y, z, hadamard h and the cnot
    
    from sympy.functions.elementary import trigonometric as sympy_trig
    from sympy.functions.elementary import exponential as sympy_exp
    import numpy as np
    from sympy import *
    
    Full working function call:
    
    # ---------------------------------------------------------------------------------------------------
    
    n = 2
    initial_state = np.zeros(2**n, dtype = complex)
    initial_state[0] = 1.0+0.0j
    gates = [['h', 0], ['x', 1], ['cnot', 0, 1]]
    final_state = symbolic_circuit_evaluation(n, gates, initial_state)
    print(final_state) # gives (1/sqrt(2))[0, 1, 1, 0] ie (1/sqrt(2))(|01> + |10>)
    
    # ---------------------------------------------------------------------------------------------------
    
    theta_1, theta_2 = symbols('theta_1 theta_2')
    n = 2
    initial_state = np.zeros(2**n, dtype = complex)
    initial_state[0] = 1.0+0.0j
    gates = [['rx', theta_1, 0], ['rx', theta_2, 1]]
    final_state = symbolic_circuit_evaluation(n, gates, initial_state)
    print(final_state) 
    
    # ---------------------------------------------------------------------------------------------------

    Outputs:
    
    final_state = the state after the circuit has acted on the initial_state (returned again as an np.array)
    
    """
    
    idn = np.array([[1, 0], [0, 1]])
    
    def rx(n, theta, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[sympy_trig.cos(theta/2), -1j*sympy_trig.sin(theta/2)], [-1j*sympy_trig.sin(theta/2), sympy_trig.cos(theta/2)]]))
        return res
    
    def rz(n, theta, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[sympy_exp.exp(-1j*theta/2), 0], [0, sympy_exp.exp(1j*theta/2)]]))
        return res
    
    def ry(n, theta, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[sympy_trig.cos(theta/2), -sympy_trig.sin(theta/2)], [sympy_trig.sin(theta/2), sympy_trig.cos(theta/2)]]))
        return res
    
    def p(n, theta, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[1, 0], [0, sympy_exp.exp(1j*theta)]]))
        return res
    
    def x(n, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[0, 1], [1, 0]]))
        return res
    
    def y(n, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[0, -1j], [1j, 0]]))
        return res
    
    def z(n, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, np.array([[1, 0], [0, -1]]))
        return res
    
    def h(n, idx):
        res = np.array([1])
        for i in range(n):
            if i != (n-1)-idx:
                res = np.kron(res, idn)
            else:
                res = np.kron(res, (1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
        return res
    
    def cnot(n, idx_control, idx_target):
        
        res = np.zeros((2**n, 2**n))
        
        for row in range(2**n):
            for col in range(2**n):
                
                row_key = bin(row)[2:]
                if len(row_key) != n:
                    row_key = (n - len(row_key))*'0' + row_key
                row_key = row_key[::-1]
                rest_row_key = ''.join([element for idx, element in enumerate(row_key) if (idx != idx_control) and (idx != idx_target)])
                    
                col_key = bin(col)[2:]
                if len(col_key) != n:
                    col_key = (n - len(col_key))*'0' + col_key
                col_key = col_key[::-1]
                rest_col_key = ''.join([element for idx, element in enumerate(col_key) if (idx != idx_control) and (idx != idx_target)])
                
                if ('1' == row_key[idx_control]) and ('1' == col_key[idx_control]) and ((('0' == row_key[idx_target]) and ('1' == col_key[idx_target])) or (('1' == row_key[idx_target]) and ('0' == col_key[idx_target]))) and (rest_row_key == rest_col_key):
                                        
                    res[row, col] = 1
                    
                if (row_key == col_key) and ('0' == row_key[idx_control]) and ('0' == col_key[idx_control]):
                                        
                    res[row, col] = 1
        
        return res
    
    final_state = initial_state
    for gate in gates:
        
        if len(gate) == 2:
            
            if gate[0] == 'x':
                final_state = np.matmul(x(n, gate[1]), final_state)     
            
            if gate[0] == 'y':
                final_state = np.matmul(y(n, gate[1]), final_state) 
                
            if gate[0] == 'z':
                final_state = np.matmul(z(n, gate[1]), final_state)  
                
            if gate[0] == 'h':
                final_state = np.matmul(h(n, gate[1]), final_state) 
        
        elif gate[0] == 'cnot':
            
            final_state = np.matmul(cnot(n, gate[1], gate[2]), final_state)
            
        else:
            
            if gate[0] == 'rx':
                final_state = np.matmul(rx(n, gate[1], gate[2]), final_state)     
            
            if gate[0] == 'ry':
                final_state = np.matmul(ry(n, gate[1], gate[2]), final_state) 
                
            if gate[0] == 'rz':
                final_state = np.matmul(rz(n, gate[1], gate[2]), final_state)  
                
            if gate[0] == 'p':
                final_state = np.matmul(p(n, gate[1], gate[2]), final_state)
                
    return final_state
