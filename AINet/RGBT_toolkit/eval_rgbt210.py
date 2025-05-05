from rgbt import RGBT210

rgbt210 = RGBT210()

# Register your tracker
rgbt210(
    tracker_name="SOWP",
    result_path="/data1/Code/luandong/WWY_code_data/ainet_tmp/ostrack_twobranch/1250/rgbt210", 
    bbox_type="ltwh")

# Evaluate multiple trackers

sr_dict = rgbt210.SR()
print(sr_dict["SOWP"][0])

pr_dict = rgbt210.PR()
print(pr_dict["SOWP"][0])

# rgbt210.draw_plot(metric_fun=rgbt210.PR)
