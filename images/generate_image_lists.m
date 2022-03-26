path = '/home/shaoche/code/coop-3dod-infra/images/car_images_train/';
frame = dir(path);
frame = frame(3:end);
fid = fopen('./train_image_list.txt','w');
for i = 1:numel(frame)
    if mod(i,100)==0
        disp([num2str(numel(frame)) '---' num2str(i)]);
    end
    frameid = frame(i).name;
    frame_number = frame(i).name;
    vehicle_set = dir([path frame_number]);
    vehicle_set = vehicle_set(3:end);
    for j = 1:numel(vehicle_set)
        vehicle_id = vehicle_set(j).name;
        vehicle_dir = dir([path frame_number '/' vehicle_id]);
        vehicle_dir = vehicle_dir(3:end);
        num_camera_for_vechile = numel(vehicle_dir)-1;
        info = load([path frame_number '/' vehicle_id '/info.txt']);
        img_set = {};
        for m = 1: num_camera_for_vechile
            img = vehicle_dir(m).name;
            if strcmp(img(end-3:end),'jpeg') == 0
                disp('error!')
                break;
            end
            img_path = [path frame_number '/' vehicle_id '/' img];
            img_set{numel(img_set)+1} =img_path;
        end
        for m = 1:numel(img_set)-1
            for n = m+1:numel(img_set)
                fprintf(fid,'%s ',img_set{m});
                fprintf(fid,'%s ',img_set{n});
                fprintf(fid,'%.10f ',info);
                fprintf(fid,'\n');
            end
        end
    end
   
end
fclose(fid);