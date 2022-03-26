path = './car_images_train_small/';
frame = dir(path);
frame = frame(3:end);
camera_correlation_cnt = zeros(6,6);
association_cnt = zeros(1,6);
for i = 1:numel(frame)
%     if mod(i,10)==0
%         disp([num2str(numel(frame)) '---' num2str(i)]);
%     end
    frameid = frame(i).name;
    vehicle_in_camera_set=[];
    frame_number = frame(i).name;
    vehicle_set = dir([path frame_number]);
    vehicle_set = vehicle_set(3:end);
    for j = 1:numel(vehicle_set)
        vehicle_id = vehicle_set(j).name;
        vehicle_dir = dir([path frame_number '/' vehicle_id]);
        vehicle_dir = vehicle_dir(3:end);
        num_camera_for_vechile = numel(vehicle_dir);
        for m = 1: num_camera_for_vechile
            imgpath = [path frame_number '/' vehicle_id '/' vehicle_dir(m).name];
            img = imread(imgpath);
        end
    end
 
end