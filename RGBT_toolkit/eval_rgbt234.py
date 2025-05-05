from rgbt import RGBT234

rgbt234 = RGBT234()

tracker = 'tracker'
# Register your tracker
rgbt234(
    tracker_name=tracker,
    result_path="/data1/Code/luandong/WWY_code_data/ainet_tmp/ostrack_twobranch/950/rgbt234", 
    bbox_type="ltwh"
)

fia_pr,_ = rgbt234.MPR(tracker)
fia_sr,_ = rgbt234.MSR(tracker)
print(tracker + "MPR \t", fia_pr)
print(tracker + "MSR \t", fia_sr)

# v42: 89.1/66.8

# # Evaluate single challenge
# pr_tc_dict = rgbt234.MPR(seqs=rgbt234.TC)
# sr_tc_dict = rgbt234.MSR(seqs=rgbt234.TC)

# # Draw a radar chart of all challenge attributes
# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MPR, filename="RGBT234_MPR_radar.png")
# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MSR)

# # Draw a curve plot
# rgbt234.draw_plot(metric_fun=rgbt234.MPR)
# rgbt234.draw_plot(metric_fun=rgbt234.MSR)