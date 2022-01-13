rm -rf convergence/*full.pdf
pdfunite `ls -v convergence/HS*.pdf` convergence/convergence_plots.pdf
pdfunite convergence/title.pdf convergence/convergence_plots.pdf HS_Convergence_Plots.pdf
