
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, Subsubsection, \
    Plot, Figure, Matrix, Alignat, Package, NoEscape, NewPage, Command
from pylatex.utils import italic
import os
import numpy as np
import sys
sys.path.append("/home/deb/Documents/MT/figures/")

baseFolder = "/home/build/FIPOPT/Data/SIF/HS/"
pFolder = baseFolder + "Problem/"
dimFolder = "/home/build/FIPOPT/include/SIF_Dimensions/Dimensions.csv"

rootFolder = "/home/build/FIPOPT/"
figFolder = "/home/build/FIPOPT/figures/"

if __name__ == '__main__':
    image_filename = os.path.join(os.path.dirname(__file__), 'kitten.jpg')

    # geometry_options = {"tmargin": "1cm", "lmargin": "10cm"}
    # doc = Document(geometr)
    # doc.packages.append(Package(r'\usepackage[outdir=./]{pdftopdf}'))
    SIF_folders = []
    for dirname in os.listdir(baseFolder):
        if dirname.endswith('.SIF') and dirname.startswith('HS'):
            SIF_folders.append(dirname)
    traj_names, nconv_traj_names = [], []
    nontraj_names, nconv_nontraj_names = [], []
    conv_names, nonconv_names = [], []
    for dirname in os.listdir(figFolder):
        if ('HS' in dirname) and (not 'HS_' in dirname):
            with open(baseFolder + dirname.split('_')[0] + ".SIF/success.txt" , 'r') as file:
                conv = int(file.readline())
            if conv:
                conv_names.append(dirname.split('_')[0])
                if dirname.endswith('Trajectory.pdf'):
                    traj_names.append(dirname.split('_')[0])
            else:
                nonconv_names.append(dirname.split('_')[0])
                if dirname.endswith('Trajectory.pdf'):
                    nconv_traj_names.append(dirname.split('_')[0])
    nontraj_names = [x for x in conv_names if x not in traj_names]
    nconv_nontraj_names = [x for x in nonconv_names if x not in nconv_traj_names]
    print(len(traj_names))
    print(len(nconv_traj_names))
    a = 0  
    doc = Document()

    doc.preamble.append(Command('title', 'FIPOPT Convergence Results for the Hock-Schittkowski Testset'))
    doc.preamble.append(Command('author', 'Jonas Hjulstad'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    doc.generate_pdf("convergence/title", clean_tex = False)

    traj_names = ['HS2', 'HS3']
    for trajname in traj_names:
        doc = Document()    
        print(trajname)
        with doc.create(Subsubsection(trajname + " (Converged)", numbering=None)):
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + trajname + "_Trajectory.pdf", width=NoEscape(r"250px"))
                kitten_pic.add_caption('Full trajectory for ' + trajname + " (converged)")
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + trajname + "_Objective_Linear.pdf", width=r"250px")
                kitten_pic.add_caption('Iteration optimality for ' + trajname + " (converged)")
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + trajname + "_Multipliers.pdf", width=r"250px")
                kitten_pic.add_caption('Inequality multipliers for ' + trajname + " (converged)")
        doc.generate_pdf("convergence/" + trajname + "_conv_traj", clean_tex=False, compiler='pdflatex', compiler_args=['-shell-escape'])
    
    for nontrajname in nontraj_names:
        print(nontrajname)
        doc = Document()
        with doc.create(Subsubsection(nontrajname + " (Converged)", numbering=None)):
            with doc.create(Figure(position='h!')) as kitten_pic:
                kitten_pic.add_image(figFolder + nontrajname + "_Objective.pdf", width='120px')
                kitten_pic.add_caption('Iteration optimality for ' + nontrajname + " (converged)")
            with doc.create(Figure(position='h!')) as kitten_pic:
                kitten_pic.add_image(figFolder + nontrajname + "_Multipliers.pdf", width='120px')
                kitten_pic.add_caption('Inequality multipliers for ' + nontrajname + " (converged)")
        doc.generate_pdf("convergence/" + nontrajname + "_conv_notraj", clean_tex=False, compiler='pdflatex', compiler_args=['-shell-escape'])
    for nconv_trajname in nconv_traj_names:
        print(nconv_trajname)
        doc = Document()
        with doc.create(Subsubsection(nconv_trajname + " (Not Converged)", numbering=None)):
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + nconv_trajname + "_Trajectory.pdf", width=NoEscape(r"250px"))
                kitten_pic.add_caption('Full trajectory for ' + nconv_trajname + " (not converged)")
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + nconv_trajname + "_Objective.pdf", width=r"250px")
                kitten_pic.add_caption('Iteration optimality for ' + nconv_trajname + " (not converged)")
            with doc.create(Figure(position='h')) as kitten_pic:
                kitten_pic.add_image(figFolder + nconv_trajname + "_Multipliers.pdf", width=r"250px")
                kitten_pic.add_caption('Inequality multipliers for ' + nconv_trajname + " (not converged)")
        doc.generate_pdf("convergence/" + nconv_trajname + "_noconv_traj", clean_tex=False, compiler='pdflatex', compiler_args=['-shell-escape'])
    
    for nconv_nontrajname in nconv_nontraj_names:
        print(nconv_nontrajname)
        doc = Document()
        with doc.create(Subsubsection(nconv_nontrajname + " (Not Converged)", numbering=None)):
            with doc.create(Figure(position='h!')) as kitten_pic:
                kitten_pic.add_image(figFolder + nconv_nontrajname + "_Objective.pdf", width='250px')
                kitten_pic.add_caption('Iteration optimality for ' + nconv_nontrajname + " (not converged)")
            with doc.create(Figure(position='h!')) as kitten_pic:
                kitten_pic.add_image(figFolder + nconv_nontrajname + "_Multipliers.pdf", width='250px')
                kitten_pic.add_caption('Inequality multipliers for ' + nconv_nontrajname + " (not converged)")
        doc.generate_pdf("convergence/" + nconv_nontrajname + "_noconv_notraj", clean_tex=False, compiler='pdflatex', compiler_args=['-shell-escape'])
