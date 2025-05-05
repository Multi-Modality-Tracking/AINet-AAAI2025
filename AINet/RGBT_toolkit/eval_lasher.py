from rgbt import LasHeR

lasher = LasHeR()

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""

# Register your tracker
tracker1, tracker2 = 'tracker1', 'tracker2'
lasher(
    tracker_name=tracker1,
    result_path="/data1/Code/luandong/WWY_code_data/ainet_tmp/ostrack_twobranch/1450/lashertestingset", 
    bbox_type="ltwh")
# lasher(
#     tracker_name=tracker2,
#     result_path="/data1/Code/luandong/WWY_code_data/Codes/sigma_fusion/ostrack_twobranch/1142/lashertestingset", 
#     bbox_type="ltwh")

# Evaluate multiple trackers
pr_dict = lasher.PR()
npr_dict = lasher.NPR()
sr_dict = lasher.SR()


print(tracker1, pr_dict[tracker1][0], npr_dict[tracker1][0], sr_dict[tracker1][0])
print('-'*30)
# print(tracker2, pr_dict[tracker2][0], npr_dict[tracker2][0], sr_dict[tracker2][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)
