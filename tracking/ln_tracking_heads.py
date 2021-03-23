import os
import os.path as osp
import subprocess
import sys

import numpy as np
import cv2

from tensorflow.keras import backend as K
# from keras.models import load_model
from tensorflow.keras.optimizers import Adam

gen_code_path = os.path.join(os.path.dirname(__file__))
print(gen_code_path)
sys.path.insert(0, gen_code_path)
sys.path.insert(0, ".")
# sys.path.insert(0, osp.join(gen_code_path, "utils"))
from detector.models.keras_ssd512 import ssd_512
from detector.keras_loss_function.keras_ssd_loss import SSDLoss
from tube_utils import iou2d, nms3dt, trackscore

# from utilstr.utils import DetectionParameters, TrackingParameters, Paths
from utilstr.utils import *
from utilstr.utils import make_if_not_exist_file, make_if_not_exist


# In this function we take the head detections and we create tracks using
# online forward and backward tracking
det_params = DetectionParameters()
track_params = TrackingParameters()
paths = Paths() # NOTE: adapt these paths to your needs


def load_detection_model(verbose=False):
    # Download the model file using the provided script
    if not osp.isfile(paths.det_model):
        if osp.isfile(paths.down_script):
            # Check the detection model has been downloaded
            print("Downloading model...")
            subprocess.call(paths.down_script)
            print("Model downloaded to {}.".format(paths.det_model))
        else:
            raise FileNotFoundError('Download script not found in {} folder.'.format(paths.down_script))

    # Clear previous models from memory.
    K.clear_session()
    model = ssd_512(image_size=(det_params.InputWidth, det_params.InputHeight, 3),
                    n_classes=1,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 128, 256, 512],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=det_params.ClipBoxes,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=det_params.NormalizeCoords,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=det_params.MinScore,
                    iou_threshold=det_params.IoUThr,
                    top_k=det_params.NKeep,
                    nms_max_output_size=det_params.NmsKeep)
    model.load_weights(paths.det_model, by_name=True)

    # Compile the model
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    if verbose:
        print(model.summary())

    return model


def generate_detections(video_path, verbose=False, outputdir=None):

    # Load the SSD detector
    model = load_detection_model()

    # Read from video file
    vcap = cv2.VideoCapture(video_path)
    if not vcap.isOpened():
        print("Could not open input video file {}".format(video_path))
    if verbose:
        print("Reading video file {}".format(video_path))

    if outputdir is None or outputdir == "":
        export = False
    else:
        export = True
        make_if_not_exist(outputdir)

    # Begin detection
    dets_dict = {}
    frame_idx = 0
    while vcap.grab():
        if verbose:
            print("Generating detections for frame %s..." % frame_idx)

        # Read frame from video
        ret, frame = vcap.retrieve()
        if ret:

            if export:
                imgname = os.path.join(outputdir, "{:06d}.jpg".format(frame_idx))
                cv2.imwrite(imgname, frame)


            # Adapt image to networks format
            org_resolution = np.flip(frame.shape[:2]) # reverse to width, height format
            frame = cv2.resize(frame, (det_params.InputWidth, det_params.InputHeight))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform prediction
            y_pred = model.predict(np.expand_dims(frame, axis=0))
            y_pred = y_pred[0]
            # Filter out detections below threshold
            y_pred = y_pred[y_pred[:, 1] > det_params.MinScore]
            # Rearrange to format [xmin, ymin, xmax, ymax, score, class]
            y_pred = y_pred[:, np.array([2, 3, 4, 5, 1, 0])]
            # Scale back detections to original frame size
            y_pred[:, :4] = y_pred[:, :4] * np.tile(org_resolution, 2) / det_params.InputWidth
            if verbose > 1:
                print(y_pred)
            dets_dict[frame_idx] = y_pred
        else:
            print("Could not read frame {}".format(frame_idx))
        frame_idx += 1
    vcap.release()

    # Save detections
    video_name = os.path.split(video_path)[-1]
    video_name = os.path.splitext(video_name)[0]
    save_pkl(paths.detections_picklefile(video_name), dets_dict)
    if verbose:
        print("Detections saved to \"%s/\"" % paths.detections_picklefile(video_name))


def process_detections_fromdetector(video_path, verbose=False, outputdir=None):
    """Load or generates detections for the given video file:
    in: path of the input video file
    out: detections: dictionary with keys the number of the frames of the shot
         Each item contains an np.array of shape: [Nx5], where N is the number of boxes for this frame
         and 5 are the bboxes: [xmin, ymin, xmax, ymax, score]
    """

    # Get video name from path
    video_name = os.path.split(video_path)[-1]
    video_name = os.path.splitext(video_name)[0]

    # load per-frame detections and save them
    # dets: matrix with rows [xmin, ymin, xmax, ymax, score, class]
    detections_pkl_file = paths.detections_picklefile(video_name)
    if not osp.isfile(detections_pkl_file):
        if verbose:
            print("Detections not found for video {}, generating...".format(video_path))
        # Generate detections
        out_dets_name = os.path.join(paths.out_detections, video_name + '.pkl')
        make_if_not_exist_file(out_dets_name)
        generate_detections(video_path, verbose=verbose, outputdir=outputdir)
    detections_pkl = load_pkl(detections_pkl_file)

    # Process detections
    detections = {f: np.empty((0, 5), dtype=np.float32) for f in range(0, len(detections_pkl))}
    for ii in range(0, len(detections_pkl)): # for all frames in clip
        nboxes = detections_pkl[ii].shape[0]
        if nboxes == 0:
            continue
        detections[ii] = detections_pkl[ii][:, :5].astype(np.float32)

    return detections


def track_forwards_backwards(detections, starting_frame, ending_frame, OUT_TRACKS=[], tracking_case='forwards', verbose=False):
    
    if tracking_case == 'forwards':
        iter_start = starting_frame
        iter_end = ending_frame
        iterator = 1
        frame_new_tracks = starting_frame # from which frame to start new tracks
        if verbose:
            print('Linking forwards') 
    elif tracking_case == 'backwards':
        iter_start = ending_frame - 1
        iter_end = starting_frame - 1
        iterator = -1
        frame_new_tracks = ending_frame # from which frame to start new tracks
        if verbose:
            print('Linking backwards') 

    # tracks is a list of tuples (frame, clsbox)
    tracks__ = [] 
    FinalTracks = []
    # main loop for all frames forwards/backwards
    for frame in range(iter_start, iter_end, iterator): 
        dets_frame = detections[frame][:, np.array([0, 1, 2, 3, 4], dtype=np.int32)] 
        # sort detections based on score
        idx_d = np.argsort(-dets_frame[:,4]) # no nms in the detections 
        dets_frame2 = dets_frame[idx_d[:], :]
        dets_frame = dets_frame2
        
        if frame == frame_new_tracks: # create new tracks 
            for i in range(dets_frame.shape[0]):
                tracks__.append([(frame, dets_frame[i, :])])
            continue

        # sort current tubes according to average score
        avgscore = [trackscore(t) for t in tracks__]
        argsort = np.argsort(-np.array(avgscore))
        tracks__ = [tracks__[i] for i in argsort]
        
        # loop over tubes 
        finished = []
        for it, t in enumerate(tracks__): 
            # compute ious between the last box of t and dets_frame
            lastbox = t[-1][1][:4]
            ious = iou2d(dets_frame[:, :4], lastbox)
            valid = np.where(ious >= track_params.IoUThr)[0]

            if valid.size > 0:
                # take the one with maximum score
                idx = valid[np.argmax(dets_frame[valid, 4])]
                tracks__[it].append((frame, dets_frame[idx, :]))
                dets_frame = np.delete(dets_frame, idx, axis=0)
            else:
                # skip
                if frame - t[-1][0] > track_params.MaxSkip:
                    finished.append(it)

        # finished tubes that are done
        for it in finished[::-1]: 
            # process in reverse order to delete them with the right index
            FinalTracks.append(tracks__[it][:])
            del tracks__[it]         

        # start new tubes 
        for i in range(dets_frame.shape[0]):
            tracks__.append([(frame, dets_frame[i, :])])

    FinalTracks += tracks__
    
    if tracking_case == 'backwards': 
        # only for backwards: reverse tracks
        FinalTracks2 = []
        for t in FinalTracks:
            t2 = []
            for tmpbox in range(len(t)-1, -1, -1):
                t2.append(t[tmpbox])
            FinalTracks2.append(t2)
            FinalTracks = []
            FinalTracks = FinalTracks2

    OUT_TRACKS += FinalTracks

    return OUT_TRACKS


def process_tracks_parameters(tracks__):
    """Post-process tracks:
    -- Discard tracks based on their score
    -- Discard tracks based on their length
    -- Do interpolation if needed between frames of tracks 
    -- Remove duplicates 
    -- Do nms3D 
    -- Returns: OUT_TRACKS: list of tuples with (track, score)
    """
    tracks = []
    for t_idx, t in enumerate(tracks__):
        score = trackscore(t)
        if score < track_params.MinScore: # discard based on score
            continue
        
        beginframe = t[0][0]
        endframe = t[-1][0]
        length = endframe - beginframe + 1
        if length < track_params.MinLength: # discard based on length
            continue

        current_track = np.empty((length, 6), dtype=np.float32)
        current_track[:, 0] = np.arange(beginframe, endframe+1)

        # do interpolation 
        for i in range(len(t)): 
            frame, box = t[i]
            current_track[frame-beginframe, 1:6] = box[:5]
            if i > 0 and t[i-1][0] != frame - 1:
                # do interpolation
                oldframe, oldbox = t[i - 1]
                for j in range(oldframe+1, frame):
                    coefold = (frame - j) / (frame - oldframe)
                    current_track[j-beginframe, 1:5] = coefold*oldbox[:4] + (1-coefold)*box[:4]
                    current_track[j-beginframe, 5] = np.average(
                        [current_track[frame - beginframe, -1], current_track[oldframe - beginframe, -1]])

        # check for duplicates 
        keep_track = True
        for kk in range(len(tracks)):
            if np.array_equal(tracks[kk][0], current_track):
                keep_track = False 
        if not keep_track:
            continue

        tracks.append((current_track, score))

    OUT_TRACKS = tracks 
    if track_params.nms3DT:
        del OUT_TRACKS
        OUT_TRACKS = []
        idx = nms3dt(tracks, track_params.nms3DT_Thr)
        OUT_TRACKS += [(tracks[i][0], tracks[i][1]) for i in idx]       

    return OUT_TRACKS


def display_tracks(video_path, tracks, verbose=False):
    # Read from video file
    vcap = cv2.VideoCapture(video_path)
    if not vcap.isOpened():
        print("Could not open input video file {}".format(video_path))
    if verbose:
        print("Reading video file {}".format(video_path))

    # Begin frame extraction
    frame_idx = 0
    colors = np.random.randint(256, size=(len(tracks), 3)) # random list of colors
    while vcap.grab():
        if verbose:
            print("Displaying track at frame %s..." % frame_idx)

        # Read frame from video
        ret, frame = vcap.retrieve()
        if ret:
            # Draw bounding boxes in tracks for this frame, if any
            for track_idx, track in enumerate(tracks):
                # Find bounding box for the current frame
                box_idx = np.where(track[0][:][:, 0] == frame_idx)[0]
                if len(box_idx) > 0:
                    box_idx = box_idx[0]
                    if verbose:
                        print(track[0][box_idx][1:5])
                    xmin = track[0][box_idx][1]
                    ymin = track[0][box_idx][2]
                    xmax = track[0][box_idx][3]
                    ymax = track[0][box_idx][4]
                    color = [int(val) for val in colors[track_idx]]
                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
                else:
                    print("No matching bbox for frame {} in track {}".format(frame_idx, track_idx))
                    continue
            # Display resulting frame
            cv2.imshow("image", frame)
            cv2.waitKey(50)
            frame_idx += 1
        else:
            print("Could not read frame {}".format(frame_idx))

    # Release video input
    print('Reached end of video.')
    vcap.release()
    cv2.destroyAllWindows()


def process_video(video_path, sanity_checks=False, verbose=False, framesdir=None):
    """Main function to create the tracks for a given video
    """
    if verbose:
        print("Processing video {}".format(video_path))

    if sanity_checks:
        # Check the video path is correct
        if not osp.isfile(video_path):
            print("VIDEO {} NOT FOUND".format(video_path))

    # Get video name from path
    video_name = os.path.split(video_path)[-1]
    video_name = os.path.splitext(video_name)[0]
    out_track_name = os.path.join(paths.out_tracks, video_name + '.pkl')
    make_if_not_exist_file(out_track_name)
    if not osp.isfile(out_track_name):
        # first load the detections and process them
        detections = process_detections_fromdetector(video_path, verbose=verbose, outputdir=framesdir)
        # tracking backwards and forwards
        if 'tracksb__' in locals(): tracksb__.clear()
        if 'tracksbf__' in locals(): tracksbf__.clear()
        if 'tracks' in locals(): tracks.clear()

        tracksb__ = track_forwards_backwards(detections, 0, len(detections),  OUT_TRACKS=[], tracking_case='backwards', verbose=verbose)
        tracksbf__ = track_forwards_backwards(detections, 0, len(detections), OUT_TRACKS=tracksb__, tracking_case='forwards', verbose=verbose)
        tracks = process_tracks_parameters(tracksbf__)

        if verbose > 1:
            print("Printing generated tracks...")
            print(tracks)

        if verbose:
            print("Finished processing tracks for video {}, now saving..".format(video_path))
        # saving
        save_pkl(out_track_name, tracks)
    else:
        print("Tracks for video {} already existing in {}".format(video_name, out_track_name))
        tracks = load_pkl(out_track_name)

    # Visualize the results
    if verbose > 2:
        display_tracks(video_path, tracks, verbose=verbose)

    return tracks


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path of the input video to process')
    parser.add_argument("--verbose", type=str2bool,
                        nargs='?', const=False, default=False,
                        help="Whether to enable verbosity of output")
    args = parser.parse_args()
    video_path = args.video_path
    verbose = args.verbose
    process_video(video_path, verbose=verbose)
