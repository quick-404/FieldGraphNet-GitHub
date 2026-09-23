"""Fix remaining small font sizes in figure script."""
with open('generate_figures.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix remaining small fonts
replacements = [
    ("ax.set_ylabel('Recall', fontsize=7, labelpad=4)", "ax.set_ylabel('Recall', fontsize=14, labelpad=6)"),
    ("ax.tick_params(axis='both', labelsize=7)\nax.yaxis.grid", "ax.tick_params(axis='both', labelsize=12)\nax.yaxis.grid"),
    ("fontsize=6, ha='center', color=fusion_color", "fontsize=10, ha='center', color=fusion_color"),
    ("fontsize=5.5, ha='center', color=gnn4id_color", "fontsize=10, ha='center', color=gnn4id_color"),
    ("ax.tick_params(axis='both', labelsize=7)\nax.text", "ax.tick_params(axis='both', labelsize=12)\nax.text"),
    ("ax.set_xlabel('Mean Recall', fontsize=7, labelpad=4)", "ax.set_xlabel('Mean Recall', fontsize=14, labelpad=6)"),
    ("ax.text(v + 0.01, i, f'{v:.3f}', fontsize=6, va='center'", "ax.text(v + 0.01, i, f'{v:.3f}', fontsize=10, va='center'"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Fixed: {old[:40]}")
    else:
        print(f"NOT FOUND: {old[:40]}")

with open('generate_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
