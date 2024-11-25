# ML-Project-2

>>1ST MAIL OF THE PROF
> Each file is in the csv format and shows the x-y center of mass coordinates of one worm.
>
> Under "Lifespan" you will find "companyDrug" and "control" subfolders, which are from one experiment where a drug turned out not to have an effect on lifespan. However, can you find a difference in behavior in the presence of the drug? Note that the lifespan recordings were made for 6 hours at a time and the csv file thus restarts the frame numbering after each 10800 frames.
>
> More lifespan data to come.
>
> Under "Optogenetics" you will find the "ATR+" and "ATR-" subfolders. ATR+ means that the optogenetic systems in the worm neurons were functional. ATR- is the control. Here, the csv files have an additional column "Light_Pulse", which is 1 when the light was turned on. Can you find worm personalities, i.e., persistent differences between worms, including in response to the light stimulus? Here is a related reference


>>>INFORMATIONS FROM SLACK LPBS
5) The data was recorded at a constant interval of 2 seconds over 30 minutes every 6 hours, so there should be 900 frames per session.


>>> WHAT I ANALYSED (might be wrong)
Key Takeaways from our Data 

1.  Frame Information:
    Each frame corresponds to 2 seconds of elapsed time.
    Therefore, 10'800 frames × 2 seconds = 21600 seconds 
    Time per frame = Session duration / Number of frames per session = 21'600 / 10'800 = 2 sec

2.  Session Structure:
    A session lasts 6 hours = 6 * 3600 = 21'600 sec
    After each session, there is a gap of 5 hours 30 minutes, during which no data is recorded.

3.  Frame Resets:
    The Frame column resets to 1 after each session of 10'800 frames.
    This means Frame alone does not provide a continuous timeline for the worm’s behavior.

4.  Session Offset:
    To account for the time elapsed from previous sessions, we add a 21600 seconds offset for each completed session.

5.  Updated Absolute Time Formula
    Absolute Time (s) = (Session Number - 1) * 21'600 + (Frame - 1) * 2
    the 1st part adds the total time of all the previous sessions
    the 2nd part converts the currents session's frame number into seconds, starting from 0 within the session


>> Handling NaN values :
I don't get it why it must be a bug with the microscope yes, but sometimes the values are just not given. for a few frames. But for a particular frame, if value of x is not given, it's neither for y and the speed thus equals 0 (must be in the skeleton code that if missing position --> set the speed to 0.