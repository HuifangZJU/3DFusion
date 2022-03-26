path = './car_images_train_full/';
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
    vehicle_set_in_camera=cell(1,6);
    for j = 1:numel(vehicle_set)
        vehicle_id = vehicle_set(j).name;
        vehicle_dir = dir([path frame_number '/' vehicle_id]);
        vehicle_dir = vehicle_dir(3:end);
        num_camera_for_vechile = numel(vehicle_dir);
        for m = 1: num_camera_for_vechile
            cam = split(vehicle_dir(m).name,'.');
            cam = cam{1}(end);
            cam = str2double(cam)+1;
            vehicle_set_in_camera{cam} = [vehicle_set_in_camera{cam} vehicle_id];
        end
    end
    cam_num_in_current_frame = numel(vehicle_set_in_camera);
    for cami = 1:cam_num_in_current_frame-1
        for camj = cami+1:cam_num_in_current_frame
            vehicle_in_cami = vehicle_set_in_camera{cami};
            vehicle_in_camj = vehicle_set_in_camera{camj};
            vehicle_in_common = intersect(vehicle_in_cami,vehicle_in_camj);
            temp = numel(vehicle_in_common);
            if temp>0
                association_cnt(temp) = association_cnt(temp)+1;
            end
            if temp==3
                disp(frameid);
                disp(cami);
                disp(camj);
            end
        end
    end
    
end