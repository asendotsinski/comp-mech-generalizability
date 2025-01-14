import subprocess

subprocess.run(
        [
            "Rscript",
            "../src_figure/logit_lens.R",
            # "../src_figure/PaperPlot.R",
            f"../results/copyVSfact/logit_lens/gpt2_full",
        ]
    )