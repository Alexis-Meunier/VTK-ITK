# vtk_itk

This project aims to study the evolution of a brain tumor over time using the tools seen in class called "ITK" and "VTK".

## Usage

To use the program, as asked by the subject, either run `uv sync` at the root of the repo, then run `uv run main.py [arguments]`, or if you already have the necessary packages on your pc, you can run it using `python main.py`.

The arguments are as follows:
- `--display {none,slice,comparison,volume}`: (default: 'comparison')
    - `none` will not display anything, and simply save a 2D plot to the `output/` folder
    - `slice` will display a 2D slice of the brain and the tumor.
    - `comparison` will display a 3D viewer with the tumor at the first timestamp in green, and the second timestamp in orange to study the change.
    - `volume` allows a viewing of the tumor with the skull a bit visible, in 3D as well.
- `--dont-plot`: Should disable the creating of 2D plots. Otherwise, running the file will create the plots at the `output/` folder

## Workload

The project was setup by alexis.meunier. Then anis.feore worked on the registration/recalage. alexis.meunier also did the segmentation and a bit of final touches. lucil.finkelstein worked on multiple visualization tools. johan.emmanuelli worked on other visualization tools.

## TODOs

We hoped to implement an automatic segmentation algorithm, but due to lack of time, as well as having multiple ongoing project, we only could implement the semi-automatic algorithm, with hardcoded seed coordinates.

## IMPORTANT NOTICE

The project was handled using GitLab. That is why you can see a `.gitlab-ci.yml` file at the root of the project. Thus all the issues, Merge Requests, and Milestones we created can not be seen.
