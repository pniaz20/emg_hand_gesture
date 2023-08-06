# emg_hand_gesture

Hand Gesture Recognition from sEMG Data.

In this project, we will train a 3D version of the EMGNet architecture to classify hand gestures based on the CAPG Myo DB-a dataset.
Training is done in PyTorch.

Look at `report.pdf` for a detailed description of the project.

## CAPG Myo Dataset

This dataset consists of 3 subdatasets collected under varying circumstances.
The gestures used in this dataset are smaller subsets of the ones used in NinaPro DB5.
The dataset is thoroughly described in this paper:

> Y. Du, W. Jin, W. Wei, Y. Hu, and W. Geng,
> "Surface EMG-Based Inter-Session Gesture Recognition Enhanced by Deep Domain Adaptation,"
> Sensors, vol. 17, no. 3, p. 458, Feb. 2017, doi: 10.3390/s17030458. [Online].
> Available: <http://dx.doi.org/10.3390/s17030458>

Characteristics of this dataset are as follows:

- 3 subdatasets: A, B and C.
- 23 subjects in total used 8 bands of 16 channel HD-sEMG wrapped around the arm.
- 8 isometric/isotonic gestures from 18 subjects in DB-a
- Same as above, from 10 subjects in DB-b
- 12 basic movements of the fingers from 10 subjects in DB-c.
- Subjects were 23 to 26 years old.
- Each electrode had a diameter of 3 mm and was arranged with an inter-electrode distance of 7.5 mm horizontally
  and 10.05 mm vertically.
- Data was band-pass filtered at 20–380 Hz and sampled at 1000 Hz, with a 16-bit A/C conversion The resulting value
  was normalized to the [−1, 1] range, corresponding to the voltage of [−2.5 mV, 2.5 mV].
- Each gesture was held for 3–10 s and repeated 10 times. To avoid fatigue, the gestures were alternated with
  a resting posture lasting 7 s.
- 2 gestures used for MVC each one trial.
- Gestures in DB-A and DB-B are the same (gestures 13 to 20 of NinaPro DB5).
- Gestures in DB-C were equivalent to gestures 1 to 12 of the NinaPro DB5.
- MVC gestures were the No. 5 and 6 of the NinaPro DB5.
- They did not perform the registration with max-force data. They provided these two max-force gestures for the
  development of gesture recognition in inter-session and inter-subject scenarios in the future.
- Every subject in DB-b contributed two recording sessions on different days, with an inter-recording interval
  greater than one week. As a result, the electrodes of the array were attached at slightly different
  positions each time.
- Each gesture in DB-a was held for 3 to 10 seconds.
- Each gesture in DB-b and DB-c was held for approximately 3 s.
- To ensure lower skin impedance in DB-b and DB-c, the skin was abraded with soft sandpaper before being
  cleansed with alcohol.
- Whereas DB-a was intended to fine-tune hyper-parameters of the recognition model, DB-b and DB-c were intended
  to evaluate intra-session and inter-session/inter-subject recognition algorithms.
- Power-line interference was removed from the sEMG signals by using a band-stop filter
  (45–55 Hz, second-order Butterworth).
- [WHY USE PREPROCESSED DATASET?] The label of each frame was assigned on the basis of the gesture performed by
  the guiding virtual hand in our acquisition software. Thus, the resulting gestures performed by the subjects may
  not perfectly match the label as a result of human reaction times. In this study, only the static part of the
  movement was used to evaluate the recognition algorithms. In other words, for each trial, the middle one-second
  window, i.e., 1000 frames of data, was used as described previously. We use the middle one second data to ensure
  that no transition movements are included in training and testing. The raw data are also available in the online
  repository.
- Table 3 of the paper summarizes which of the 23 subjects were used in which experiment.
- Their own gesture recognition method is based on instantaneous 8 x 16 image, it is only an image classification
  problem, not a time series problem at all. They are completely disregarding the temporal dependency of the data,
  claiming that with HD-sEMG that is unnecessary.
- To prevent overfitting of small training set in some experiments, the ConvNet was initialized by pre-training
  using all available data when appropriate.
- A majority voting scheme was used when two or more frames were available. Using this scheme,
a window of sEMG signals was labeled with the class that received the most votes.
