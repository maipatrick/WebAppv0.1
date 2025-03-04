from Sports2D.Sports2D import process, DEFAULT_CONFIG

# Modify the default configuration as needed
config = DEFAULT_CONFIG.copy()
config['project']['video_input'] = [r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\fp22_m505_left 2.MOV"]
config['process']['result_dir'] = 'fp22_m505_left 2'
config['process']['show_realtime_results'] = False  # Disable real-time display
config['process']['save_vid'] = False
config['process']['save_img'] = False
config['process']['save_pose'] = True
config['process']['calculate_angles'] = True
config['process']['save_angles'] = True
config['process']['multi_person'] = True
#px_to_meters_conversion
config['px_to_meters_conversion']['to_meters'] = True
config['px_to_meters_conversion']['make_c3d'] = False
# filter
config['post-processing']['butterworth']['cut_off_frequency'] = 6
config['post-processing']['show_graphs']= False

# Call the process function with the modified configuration
process(config)