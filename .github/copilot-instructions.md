# Copilot Instructions for machine-learning-zoomcamp-homework

## Project Overview
This repository contains Jupyter notebooks for learning and practicing core machine learning concepts, with a focus on hands-on exercises. The main files are:
- `Introduction-to-numpy.ipynb`: Numpy basics and array operations
- `linear-algebra.ipynb`: Linear algebra concepts and vector/matrix operations

## Architecture & Workflow
- All code is written in Jupyter notebooks using Python and numpy.
- Each notebook is self-contained and does not import code from other files.
- There are no custom Python modules or package structure; all logic is inline in notebook cells.
- The workspace is flat, with notebooks and README at the root.

## Developer Workflow
- To run or test code, execute cells in the notebook interactively.
- No build or test scripts are present; validation is manual via notebook execution.
- Use numpy for all array, vector, and matrix operations. Avoid using pandas, sklearn, or other libraries unless explicitly instructed.
- Follow the style and conventions shown in existing cells (e.g., variable naming: `u`, `v`, `V`, `a`).

## Patterns & Conventions
- Mathematical operations are performed using numpy functions and methods (e.g., `np.array`, `u.dot(v)`).
- Explanations and markdown cells are used to clarify concepts and document steps.
- Each notebook cell should be self-explanatory and runnable independently.
- When adding new content, prefer concise code and clear markdown explanations.

## Examples
- Vector creation: `u = np.array([2, 4, 6, 8])`
- Matrix creation: `V = np.array([[1, 3, 5, 7], [2, 4, 6, 8], [0, 1, 0, 2]])`
- Dot product: `u.dot(v)`

## External Dependencies
- Only numpy is required. No additional dependencies are installed or imported.

## Key Files
- `Introduction-to-numpy.ipynb`: Numpy basics
- `linear-algebra.ipynb`: Linear algebra and vector/matrix operations
- `README.md`: Project overview

---
If any conventions or workflows are unclear, please provide feedback so this guide can be improved.