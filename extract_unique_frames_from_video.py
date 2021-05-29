
import cv2,os,time

video_name = "PigsBarnStraw4.MOV"
video = cv2.VideoCapture(video_name)

try:
    # creating a folder named data
    if not os.path.exists(video_name.split(".")[0]):
        os.makedirs(video_name.split(".")[0])

except OSError:
    print ('Error: Creating directory of data')

currentframe = 0

#Check the version of opencv you are using
# Find OpenCV version

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
else:    
    fps = int(video.get(cv2.CAP_PROP_FPS))

#The FPS generated could be decimal as well. We get its greatest interger function by casting to int().

print('The FPS of the video is ', str(fps))
time_start=time.time()
    
while(True):
      
    # reading from frame
    ret,frame = video.read()
  
    if ret:
        # if video is still left continue creating images
        name = './'+video_name.split(".")[0]+'/frame' + str(currentframe) + '.jpg'
        
        if currentframe % fps == 1: 
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
            currentframe += 1
            continue
        else:
            currentframe += 1
            continue
  
        # increasing counter so that it will
        # show how many frames are created
        
    else:
        break

time_end=time.time()
print('Total time taken is ', time_end-time_start)    
print('Total frames are ', currentframe)

#Release all space and windows once done
video.release()
cv2.destroyAllWindows()