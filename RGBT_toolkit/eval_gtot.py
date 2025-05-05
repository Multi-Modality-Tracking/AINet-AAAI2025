from rgbt import GTOT
from rgbt.utils import RGBT_start,RGBT_end

RGBT_start()
gtot = GTOT()

# Register your tracker
gtot(
    tracker_name="JMMAC",
    result_path="/data1/Code/luandong/WWY_code_data/ainet_tmp/ostrack_twobranch/1150/gtot", 
    bbox_type="ltwh")


# Evaluate multiple trackers
pr_dict = gtot.MPR()
print(pr_dict["JMMAC"][0])

# Evaluate single tracker
jmmac_sr,_ = gtot.MSR("JMMAC")
print("JMMAC MSR:\t", jmmac_sr)

# v42: 91.9/75.7

# Draw a curve plot
# gtot.draw_plot(metric_fun=gtot.MPR)
# gtot.draw_plot(metric_fun=gtot.MSR)

RGBT_end()