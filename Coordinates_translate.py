"""
Bunker and Jensen 2nd edition p.203 - 220

A2B分子のLandau座標系から分子固定Cartisianへの座標変換
A_1 - B_3 - A_2
"""

import numpy as np
from scipy.optimize import newton_krylov

#molecular parameter
r_e: float = 0.957
alpha_e: float = 104.5
m_H:float = 1
m_O:float = 16
mass: list = [m_H, m_H, m_O]

#displacement
r_1: float = 1.4623
r_2: float = -0.0118
alpha: float = -1.1

def PRINT_MAT(mat):
    for ii in range(0, len(mat)):
        print("atom", str(ii), "Cartisian", end=',\t')
        for jj in range(0, len(mat[ii])):
            print(round(mat[ii][jj], 6), end=',\t')
        print()
    print()

def Init_to_SF(m_A, m_B, r_e, alpha_e, r_1, r_2, alpha):
    
    #B原子を原点にした座標系
    XX_1: float = - (r_e + r_1) * np.cos((alpha_e + alpha)/2 *np.pi/180)
    YY_1: float = 0
    ZZ_1: float = - (r_e + r_1) * np.sin((alpha_e + alpha)/2 *np.pi/180)
    
    XX_2: float = - (r_e + r_2) * np.cos((alpha_e + alpha)/2 *np.pi/180)
    YY_2: float = 0
    ZZ_2: float = (r_e + r_2) * np.sin((alpha_e + alpha)/2 *np.pi/180)
    
    XX_3: float = 0
    YY_3: float = 0
    ZZ_3: float = 0

    Coords_init: list = [[XX_1, YY_1, ZZ_1], 
                         [XX_2, YY_2, ZZ_2], 
                         [XX_3, YY_3, ZZ_3]]
    
    #center of mass
    X_CM: float = (m_A*XX_1 + m_A*XX_2 + m_B*XX_3)/(m_A + m_A + m_B)
    Y_CM: float = (m_A*YY_1 + m_A*YY_2 + m_B*YY_3)/(m_A + m_A + m_B)
    Z_CM: float = (m_A*ZZ_1 + m_A*ZZ_2 + m_B*ZZ_3)/(m_A + m_A + m_B)

    Coords_CM: list = [X_CM, Y_CM, Z_CM]

    #重心を原点にする空間固定座標系に移動
    Coords_SF: list = []
    for ii in range(0, len(Coords_init)):
        Coord_SF: list = []
        for jj in range(0, len(Coords_init[ii])):
            Coord_SF.append(Coords_init[ii][jj] - Coords_CM[jj])
        Coords_SF.append(Coord_SF)
    
    return Coords_SF

def SF_to_MF(coords_SF):
    #space-fixed -> mol-fixedの回転座標変換は今のアルゴリズムでは不要なので省略
    Coords_MF: list = []
    for ii in range(0, len(coords_SF)):
        Coord_MF: list = []
        for jj in range(0, len(coords_SF[ii])):
            Coord_MF.append(coords_SF[ii][jj])
        Coords_MF.append(Coord_MF)
    return Coords_MF

def MF_to_Eckart_coeff(m_A, m_B, r_e, alpha_e, r_1, r_2, alpha):
    #Eckart座標への座標変換
    #Eckart equation
    Eckart_Coeff: list = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],]
    Coord_deform_MF: list = SF_to_MF(Init_to_SF(m_H, m_O, r_e, alpha_e, r_1, r_2, alpha))
    Coord_eq_MF: list = SF_to_MF(Init_to_SF(m_H, m_O, r_e, alpha_e, 0., 0., 0.))
    for ii in range(0, len(Eckart_Coeff)):
        for jj in range(0, len(Eckart_Coeff)):
            for nn in range (0, len(mass)):
                Eckart_Coeff[ii][jj] += mass[nn]*Coord_eq_MF[nn][jj]*Coord_deform_MF[nn][ii]

    return Eckart_Coeff

def rotation_matrix_rad(x):
    #x = [phi, theta, chi]
    cosp: float = np.cos(x[0])
    cost: float = np.cos(x[1])
    cosc: float = np.cos(x[2])
    sinp: float = np.sin(x[0])
    sint: float = np.sin(x[1])
    sinc: float = np.sin(x[2])
    lambdaxx: float = + cost*cosp*cosc - sinp*sinc
    lambdaxe: float = + cost*sinp*cosc + cosp*sinc
    lambdaxz: float = - sint*cosc
    lambdayx: float = - cost*cosp*sinc - sinp*cosc
    lambdaye: float = - cost*sinp*sinc + cosp*cosc
    lambdayz: float = + sint*sinc
    lambdazx: float = + sint*cosp
    lambdaze: float = + sint*sinp
    lambdazz: float = + cost
    rotation_matrix: np.ndarry = np.array([[lambdaxx, lambdaxe, lambdaxz],
                                           [lambdayx, lambdaye, lambdayz],
                                           [lambdazx, lambdaze, lambdazz]])
    return rotation_matrix

def Eckart_equation(x):
    Eckart_Coeff: list = MF_to_Eckart_coeff(m_H, m_O, r_e, alpha_e, r_1, r_2, alpha)
    rot_mat: list = rotation_matrix_rad(x)
    function: list = [Eckart_Coeff[0][0]*rot_mat[1][0] + Eckart_Coeff[1][0]*rot_mat[1][1] + Eckart_Coeff[2][0]*rot_mat[1][2] - Eckart_Coeff[0][1]*rot_mat[0][0] - Eckart_Coeff[1][1]*rot_mat[0][1] - Eckart_Coeff[2][1]*rot_mat[0][2], 
                Eckart_Coeff[0][1]*rot_mat[2][0] + Eckart_Coeff[1][1]*rot_mat[2][1] + Eckart_Coeff[2][1]*rot_mat[2][2] - Eckart_Coeff[0][2]*rot_mat[1][0] - Eckart_Coeff[1][2]*rot_mat[1][1] - Eckart_Coeff[2][2]*rot_mat[1][2], 
                Eckart_Coeff[0][2]*rot_mat[0][0] + Eckart_Coeff[1][2]*rot_mat[0][1] + Eckart_Coeff[2][2]*rot_mat[0][2] - Eckart_Coeff[0][0]*rot_mat[2][0] - Eckart_Coeff[1][0]*rot_mat[1][2] - Eckart_Coeff[2][0]*rot_mat[2][2]
                ]
    return function

def solve_Eckart_eq(function, init):
    sol:list = newton_krylov(Eckart_equation, init, method='lgmres')
    sol_deg: list = [0, 0, 0]
    sol_deg[0] = (sol[0])*180/np.pi
    sol_deg[1] = (sol[1])*180/np.pi
    sol_deg[2] = (sol[2])*180/np.pi
    print("Eular angle: mol-fixed -> Eckart")
    print(sol_deg, '\n')
    return sol
    
def rotation_MF_Eckart(Coords_MF, rotation_matrix):
    Coords_MF_Cartisian: list = []
    for ii in range(0, len(Coords_MF)):
        Coord_MF_Cartisian: list = []
        for jj in range(0, 3):
            element: float = 0
            for kk in range(0, 3):
                element += rotation_matrix[jj][kk]*Coords_MF[ii][kk]
            Coord_MF_Cartisian.append(element)
        Coords_MF_Cartisian.append(Coord_MF_Cartisian)

    return Coords_MF_Cartisian
    
def main():

    deform_MF: list = SF_to_MF(Init_to_SF(m_H, m_O, r_e, alpha_e, r_1, r_2, alpha))
    eq_MF: list = SF_to_MF(Init_to_SF(m_H, m_O, r_e, alpha_e, 0, 0, 0))
    print("deformation molecule, molecular fixed coordinates: ")
    PRINT_MAT(deform_MF)
    print("equiliblium molcule, molecular fixed coordinate: ")
    PRINT_MAT(eq_MF)
    
    init: list = [0*np.pi/180, 0*np.pi/180, 0*np.pi/180]
    rot_MF_Eckart: list = solve_Eckart_eq(Eckart_equation(init), init)
    Coords_Eckart_Cartisian: list = rotation_MF_Eckart(deform_MF, rotation_matrix_rad(rot_MF_Eckart))
    print("Cartisian for each atoms in Eckart coordinates: ")
    PRINT_MAT(Coords_Eckart_Cartisian)

    delta_Eckart_Cartisian = []
    for ii in range (0, len(Coords_Eckart_Cartisian)):
        delta = []
        for jj in range (0, len(Coords_Eckart_Cartisian[ii])):
            delta.append(Coords_Eckart_Cartisian[ii][jj] - eq_MF[ii][jj])
        delta_Eckart_Cartisian.append(delta)
       
    print("deformation delta in Eckart coordinates: ")
    PRINT_MAT(delta_Eckart_Cartisian)
    
    return

main()
