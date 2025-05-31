import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import linprog

# --- Configuration de la Page Streamlit ---
st.set_page_config(page_title="Résolution Graphique - PL", layout="centered")
st.title("Résolution Graphique de Programmation Linéaire")


# --- VALEURS PAR DÉFAUT ET FONCTION DE RESTAURATION ---
DEFAULTS = {
    "problem_type": "Maximiser",
    "obj_func": "3x1 + 2x2",
    "n_contraintes": 2,
    "non_negative": True,
    "constraints": ["x1 + x2 <= 6", "2x1 + x2 <= 8"]
}

def restore_defaults():
    """Fonction de rappel pour restaurer tous les widgets à leurs valeurs par défaut."""
    st.session_state.problem_type = DEFAULTS["problem_type"]
    st.session_state.obj_func = DEFAULTS["obj_func"]
    st.session_state.n_contraintes = DEFAULTS["n_contraintes"]
    st.session_state.non_negative = DEFAULTS["non_negative"]
    
    # Restaurer les champs de contraintes visibles
    for i in range(DEFAULTS["n_contraintes"]):
        st.session_state[f"contrainte_{i}"] = DEFAULTS["constraints"][i]
    
    # Nettoyer les anciens champs de contraintes s'il y en avait plus de 2
    for i in range(DEFAULTS["n_contraintes"], 10): # 10 est la valeur max du number_input
        if f"contrainte_{i}" in st.session_state:
            del st.session_state[f"contrainte_{i}"]


# --- FONCTIONS D'ANALYSE (PARSING) ---
def parse_expression(expression_str):
    expression_str = expression_str.strip().lower().replace(" ", "")
    coeffs = {'1': 0.0, '2': 0.0}
    if not expression_str: return [0.0, 0.0]
    if not expression_str.startswith(('+', '-')): expression_str = '+' + expression_str
    terms = re.findall(r"([+-])(\d*\.?\d*)?\*?x([12])", expression_str)
    for sign, coeff_str, var_index in terms:
        coeff = float(coeff_str) if coeff_str else 1.0
        if sign == '-': coeff = -coeff
        coeffs[var_index] += coeff
    return [coeffs['1'], coeffs['2']]

def parse_constraint(constraint_str):
    constraint_str = constraint_str.strip()
    if not constraint_str: return None
    match = re.search(r"(<=|>=|=)", constraint_str)
    if not match:
        st.error(f"Contrainte invalide: '{constraint_str}'. Signe (<=, >=, =) manquant.")
        return None
    sense = match.group(1)
    lhs, rhs_str = constraint_str.split(sense)
    try:
        b = float(rhs_str.strip())
    except ValueError:
        st.error(f"Valeur de droite invalide: '{rhs_str.strip()}'")
        return None
    coeffs = parse_expression(lhs)
    return (coeffs, b, sense)


# --- INTERFACE UTILISATEUR STREAMLIT ---

# On s'assure que les valeurs par défaut sont dans le session_state au premier lancement
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
for i, c_val in enumerate(DEFAULTS["constraints"]):
    if f"contrainte_{i}" not in st.session_state:
        st.session_state[f"contrainte_{i}"] = c_val

problem_type = st.radio("Type de problème", ["Maximiser", "Minimiser"], key="problem_type")

st.markdown("### Fonction Objectif")
obj_func_str_input = st.text_input("Entrez la fonction objectif", key="obj_func")
c_obj = parse_expression(obj_func_str_input)
st.latex(f"{st.session_state.problem_type} \\ Z = {st.session_state.obj_func}")

st.markdown("### Contraintes")
n_contraintes = st.number_input("Nombre de contraintes", min_value=1, max_value=10, step=1, key="n_contraintes")
contraintes = []

for i in range(n_contraintes):
    constraint_input_str = st.text_input(f"Contrainte {i+1}", key=f"contrainte_{i}")
    if constraint_input_str:
        st.latex(constraint_input_str.replace("<=", r" \leq ").replace(">=", r" \geq "))
        parsed_constraint = parse_constraint(constraint_input_str)
        if parsed_constraint:
            contraintes.append(parsed_constraint)

non_negative = st.checkbox("Inclure les contraintes de non-négativité (x₁, x₂ ≥ 0)", key="non_negative")

# --- BOUTONS D'ACTION ---
col1, col2 = st.columns([3, 2]) # Le premier bouton est plus large

with col1:
    solve_button = st.button("Résoudre et Afficher le Graphique", type="primary", use_container_width=True)
with col2:
    st.button("Restaurer les valeurs", on_click=restore_defaults, use_container_width=True)


# --- LOGIQUE DE RÉSOLUTION ET D'AFFICHAGE ---

if solve_button:
    if not contraintes or not any(c != 0 for c in c_obj):
        st.warning("Veuillez définir une fonction objectif et des contraintes valides.")
    else:
        # Configuration du graphique
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel("$x_1$", fontsize=12)
        ax.set_ylabel("$x_2$", fontsize=12)
        ax.set_title("Visualisation du Problème de Programmation Linéaire", fontsize=14)

        plot_range = 20
        d_grid = np.linspace(-plot_range, plot_range, 400)
        x_grid, y_grid = np.meshgrid(d_grid, d_grid)

        # Coloriage de la zone réalisable
        feasible_mask = np.full(x_grid.shape, True)
        all_constraints_for_fill = list(contraintes)
        if non_negative:
            all_constraints_for_fill.append(([1, 0], 0, ">="))
            all_constraints_for_fill.append(([0, 1], 0, ">="))

        for coeffs, b, sense in all_constraints_for_fill:
            a1, a2 = coeffs
            expr = a1 * x_grid + a2 * y_grid
            if sense == '<=': feasible_mask &= (expr <= b + 1e-9)
            elif sense == '>=': feasible_mask &= (expr >= b - 1e-9)
            elif sense == '=': feasible_mask &= np.isclose(expr, b)
        
        if np.any(feasible_mask):
            ax.contourf(x_grid, y_grid, feasible_mask, levels=[0.5, 1.5], colors=['green'], alpha=0.3)
            ax.plot([], [], color='green', alpha=0.3, linewidth=10, label='Région Réalisable')
        
        # Dessin des lignes et calcul des limites du graphique
        d_lines = np.linspace(-plot_range, plot_range, 400)
        lines = []
        max_coords = [10]
        for coeffs, b, sense in contraintes:
            a1, a2 = coeffs
            label = f"{a1 if a1!=1 and a1!=0 else ''}{'x₁' if a1!=0 else ''}{'+' if a2>0 else ''}{a2 if a2!=1 and a2!=0 else ''}{'x₂' if a2!=0 else ''} {sense} {b}".replace("+ -","- ")
            if a2 != 0:
                y_vals = (b - a1 * d_lines) / a2
                ax.plot(d_lines, y_vals, label=label)
                if a1 != 0: max_coords.append(abs(b/a1))
                max_coords.append(abs(b/a2))
            elif a1 != 0:
                x_val = b / a1
                ax.axvline(x=x_val, label=label)
                max_coords.append(abs(x_val))
            lines.append((coeffs, b))
        
        if non_negative:
            lines.extend([([1, 0], 0), ([0, 1], 0)])

        # Calcul des sommets et affichage
        vertices = set()
        if len(lines) >= 2:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    A = np.array([lines[i][0], lines[j][0]])
                    B = np.array([lines[i][1], lines[j][1]])
                    try:
                        sol = np.linalg.solve(A, B)
                        is_feasible = True
                        for c, b, s in all_constraints_for_fill:
                            val = np.dot(c, sol)
                            if not ((s == '<=' and val <= b + 1e-5) or (s == '>=' and val >= b - 1e-5) or (s == '=' and np.isclose(val, b))):
                                is_feasible = False
                                break
                        if is_feasible: vertices.add(tuple(np.round(sol, 2)))
                    except np.linalg.LinAlgError: continue
        
        # Affichage des informations sur les sommets et la région
        if not np.any(feasible_mask):
            st.warning("La région réalisable est **vide**.")
        else:
            if vertices:
                st.markdown("#### Valeurs de la fonction objectif aux sommets :")
                for i, (vx, vy) in enumerate(sorted(list(vertices))):
                    z_val = np.dot(c_obj, [vx, vy])
                    ax.plot(vx, vy, 'o', color='red', markersize=8, zorder=5)
                    ax.text(vx + 0.1, vy + 0.1, f"({vx}, {vy})", fontsize=9, zorder=6)
                    st.markdown(f"- Au point ({vx}, {vy}) : $Z = {z_val:.2f}$")
            
            top = feasible_mask[-1, :].any(); right = feasible_mask[:, -1].any()
            bottom = feasible_mask[0, :].any(); left = feasible_mask[:, 0].any()
            if top or right or bottom or left: st.info("La région réalisable est **non bornée**.")
            else: st.info("La région réalisable est **bornée**.")

        view_limit = max(max_coords) * 1.2
        ax.set_xlim(0 if non_negative else -view_limit/2, view_limit)
        ax.set_ylim(0 if non_negative else -view_limit/2, view_limit)
        ax.legend()
        st.pyplot(fig)

        # --- Résolution avec SciPy (linprog) ---
        A_ub_list, b_ub_list, A_eq_list, b_eq_list = [], [], [], []
        for c, b, s in contraintes:
            if s == '<=': 
                A_ub_list.append(c)
                b_ub_list.append(b)
            elif s == '>=': 
                A_ub_list.append([-x for x in c])
                b_ub_list.append(-b)
            elif s == '=': 
                A_eq_list.append(c)
                b_eq_list.append(b)
        
        # Convert lists to NumPy arrays, handling empty cases correctly
        A_ub = np.array(A_ub_list) if A_ub_list else None
        b_ub = np.array(b_ub_list) if b_ub_list else None
        A_eq = np.array(A_eq_list) if A_eq_list else None
        b_eq = np.array(b_eq_list) if b_eq_list else None

        bounds = [(0, None), (0, None)] if non_negative else [(None, None), (None, None)]
        
        # Ensure c_vector is always a numpy array for consistency
        c_vector = np.array([-v for v in c_obj]) if st.session_state.problem_type == "Maximiser" else np.array(c_obj)
        
        # Debugging: Print shapes before linprog
        # st.write(f"c_vector shape: {c_vector.shape}")
        # if A_ub is not None: st.write(f"A_ub shape: {A_ub.shape}")
        # if b_ub is not None: st.write(f"b_ub shape: {b_ub.shape}")
        # if A_eq is not None: st.write(f"A_eq shape: {A_eq.shape}")
        # if b_eq is not None: st.write(f"b_eq shape: {b_eq.shape}")

        try:
            res = linprog(c=c_vector, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                z_opt = np.dot(c_obj, res.x)
                st.success(f"""**Solution optimale trouvée :**\n
- **$x_1 = {res.x[0]:.2f}$**\n- **$x_2 = {res.x[1]:.2f}$**\n- **Valeur de Z = {z_opt:.2f}**""")
            else:
                st.error(f"Le solveur n'a pas trouvé de solution optimale finie. Message : {res.message}")
        except ValueError as e:
            st.error(f"Erreur lors de l'appel à linprog : {e}. Veuillez vérifier vos entrées.") 