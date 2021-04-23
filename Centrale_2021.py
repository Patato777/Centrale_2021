import math
import numpy as np

#I. Géométrie
#Q1.
def vec(A, B):
    return B-A

#Q2.
def ps(v1, v2):
    return np.inner(v1, v2)

#Q3.
def norme(v):
    return math.sqrt(ps(v, v))

#Q4.
def unitaire(v):
    return v/norme(v)

#Q5.
def pt(r, t):
    assert t >= 0
    (S, u) = r
    return S + t * u

def dir(A, B):
    return unitaire(vec(A, B))

def ra(A, B):
    return A, dir(A, B)

#Q6.
def sp(A, B):
    return A, norme(vec(A, B))

#Q8.
def intersection(r, s):
    CA = vec(s[0], r[0])
    det = 4 * (ps(r[1], CA) ** 2 - norme(CA) ** 2 + s[1] ** 2)
    if det >= 0:
        t = min((-2 * ps(r[1], CA) - math.sqrt(det)) / 2, (-2 * ps(r[1], CA) + math.sqrt(det)))
        if t >= 0:
            return pt(r, t), t

#II. Optique
noir = np.array([0., 0., 0.])
blanc = np.array([1. , 1., 1.])

    #II.A - Visibilité
#Q10.
def au_dessus(s, P, src):
    return (np.round(intersection(ra(src, P), s)[0] - P,7)==[0.,0.,0.]).all()

#Q11.
def visible(obj, j, P, src):
    return au_dessus(obj[j], P, src) and len([i for i in range(len(obj)) if i != j and intersection(ra(src, P), obj[i]) != None]) == 0

    #II.B - Diffusion
#Q12.
def couleur_diffusée(r, Cs, N, kd):
    costheta = abs(ps(N, r[1]))
    return (kd * Cs) * costheta

    #II.C - Réflexion
#Q13.
def rayon_réfléchi(s, P, src):
    N = unitaire(vec(s[0], P))
    r = ra(src, P)
    costheta = abs(ps(N, r[1]))
    return (P, unitaire(r[1] + 2 * costheta * norme(vec(src, P))))

#IV. Lancer de rayons
Objet = [(np.array([0.,0.,-10]), 5), (np.array([5.,5.,-6.]), 3)]
KdObj = [np.array([0.5,0.,0.2]), np.array([0.,0.,1.])]
Source = [np.array([0.,0.,0.])]
ColSrc = [np.array([1.,1.,0.5])]
Delta = 10
N = 200

    #IV.A - Écran
#Q18.
def grille(i, j):
    return np.array([(Delta / N) * (i - N/2 + 1/2), (Delta / N) * (j - N/2 + 1/2), 0.])

#Q19.
def rayon_écran(omega, i, j):
    return ra(omega, grille(i, j))

    #IV.B - Couleur d'un pixel
#Q20.
def interception(r):
    intersecte = [(*intersection(r, s), i) for i, s in enumerate(Objet) if intersection(r, s) != None]
    if len(intersecte) > 0:
        point = min(intersecte, key=lambda i: i[1])
        return point[0], point[2]

#Q21.
def couleur_diffusion(P, j):
    N = unitaire(vec(Objet[j][0], P))
    return sum([couleur_diffusée(ra(src,P), Cs, N, KdObj[j]) for src, Cs in zip(Source, ColSrc) if visible(Objet, j, P, src)])

    #IV.C - Constitution de l'image
#Q22.
def lancer(omega, fond):
    image = np.array([[fond] * N] * N)
    for i in range(N):
        for j in range(N):
            point = interception(rayon_écran(omega, i, j))
            if point == None:
                image[i, j] = fond
            else:
                image[i, j] = couleur_diffusion(*point)
    return image

#V. Améliorations
    #V.A - Prise en compte de la réflexion
#Q25.
def réflexions(r, rmax):
    point = interception(r)
    if point == None or rmax == 0:
        return list()
    else:
        r = rayon_réfléchi(Objet[point[1]], point[0], r[0])
        return [point] + réflexions(r, rmax - 1)

KrObj = [0.7,0.4]
#Q26.
def couleur_perçue(r, rmax, fond):
    rebonds = réflexions(r, rmax)
    if rebonds == list():
        return fond
    else:
        col = noir
        for p in reversed(rebonds):
            col = couleur_diffusion(*p) + KrObj[p[1]] * col
        return col

#Q27.
def lancer_complet(omega, fond, rmax):
    image = np.array([[fond] * N] * N)
    for i in range(N):
        for j in range(N):
            image[i, j] = couleur_perçue(rayon_écran(omega, i, j), rmax, fond)
    return image

    #V.B - Une optimisation
IdObj = list()
IdSrc = list()

#Q29.
risque = list()

def table_risque(risque):
    return [[[IdObj.index(r[2]) for r in risque if IdObj.index(r[0]) == i and IdSrc.index(r[1]) == j] for j in range(len(Source))] for i in range(len(Objet))]

#Q30.
TableRisque = table_risque(risque)

def visible_opt(j, k, P):
    return au_dessus(Objet[k], P, Source[j]) and len([i for i in TableRisque[k][j] if i != k and intersection(ra(Source[j], P), Objet[i]) != None]) == 0
