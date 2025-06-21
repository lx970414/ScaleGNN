import os
import csv

class Logger:
    def __init__(self, outdir, alpha_dim=None, mk_dim=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.alpha_log = open(os.path.join(outdir, "alpha_history.csv"), 'w', newline='')
        self.mk_log = open(os.path.join(outdir, "mk_history.csv"), 'w', newline='')
        self.alpha_writer = csv.writer(self.alpha_log)
        self.mk_writer = csv.writer(self.mk_log)
        # 动态支持列数
        if alpha_dim is not None:
            self.alpha_writer.writerow(['epoch'] + [f'alpha_{i}' for i in range(alpha_dim)])
        if mk_dim is not None:
            self.mk_writer.writerow(['epoch'] + [f'mk_{i}' for i in range(mk_dim)])

    def log(self, epoch, alpha_list=None, mk_list=None):
        if alpha_list is not None:
            self.alpha_writer.writerow([epoch] + [float(a) for a in alpha_list])
            self.alpha_log.flush()
        if mk_list is not None:
            self.mk_writer.writerow([epoch] + [int(m) for m in mk_list])
            self.mk_log.flush()

    def close(self):
        self.alpha_log.close()
        self.mk_log.close()

