#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>


class Camera {
private:
  cv::VideoCapture _device;
  cv::Size_<int>  _size;
  cv::Mat _frame;

public:
  Camera(int device=0): _device(device), _size(640, 480)
  {
    _device.set(CV_CAP_PROP_FRAME_WIDTH, _size.width);
    _device.set(CV_CAP_PROP_FRAME_HEIGHT, _size.height);
  }

  bool capture()
  {
    if(!_device.isOpened())
      return false;

    _device >> _frame;
    return true;
  }

  cv::Mat frame()
  {
    return _frame.clone();
  }
};


class Window {
private:
  std::string _tag;
  
public:
  Window(std::string tag) : _tag(tag){}

  void show(){ cv::namedWindow(_tag); }

  void update_image(cv::Mat image){ cv::imshow(_tag, image); }

  bool wait_key(int time=30)
  {
    if(cv::waitKey(30) >= 0)
      return true;
    
    return false;
  }

  void move(cv::Point point)
  {
    // opencv2.3 don't implement cv::moveWindow :p
    cvMoveWindow(_tag.c_str(), point.x, point.y);
  }
};


class FaceDetector {
public:
  std::vector<cv::Rect> detect(cv::Mat &image)
  {
    std::vector<cv::Rect> faces;  
    cv::Mat gray;
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    std::string cascade_file = "./haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascade_file))
      return faces;

    cascade.detectMultiScale(gray,
                             faces,
                             1.1,
                             2,
                             CV_HAAR_SCALE_IMAGE,
                             cv::Size(30,30));

    return faces;
  }

  void draw(cv::Mat &image, std::vector<cv::Rect> faces)
  {
    std::vector<cv::Rect>::const_iterator r;
    for(r=faces.begin(); r!=faces.end(); ++r){
      cv::rectangle(image, 
                    cv::Point(r->x, r->y),
                    cv::Point(r->x+r->width, r->y+r->height),
                    cv::Scalar(0, 200, 0),
                    3,
                    4);
      cv::circle(image, 
                 cv::Point(cv::saturate_cast<int>(r->x+r->width*0.5),
                           cv::saturate_cast<int>(r->y+r->height*0.5)),
                 5,
                 cv::Scalar(0, 0, 200),
                 8,
                 8);
                
    }    
  }

  void detect_and_draw(cv::Mat &image)
  {
    std::vector<cv::Rect> faces;
    faces = detect(image);
    draw(image, faces);
  }
  
};


class InputTracker {
protected:
  bool _enabled;

public:
  InputTracker() : _enabled(true) {}

  bool is_enable(){ return _enabled; }

  void enable(bool t){ _enabled = t; }

  virtual void send_point(cv::Point point) = 0;

};


class ForceTracker : public InputTracker {
private:
  void (*_force_begin_action)(ForceTracker *sender);
  void (*_force_move_action)(ForceTracker *sender);
  void *_user_data;
  cv::Point _initial_point;
  cv::Point _current_point;
  int _threshold;
  bool _tracking;

  void force_begin()
  {
    if(_enabled && _force_begin_action)
      _force_begin_action(this);
  }

  void force_move()
  {
    if(_enabled && _force_move_action)
      _force_move_action(this);
  }


public:
  ForceTracker() : 
    _force_begin_action(NULL)
    , _force_move_action(NULL)
    , _user_data(NULL)
    , _initial_point(cv::Point(0,0))
    , _current_point(cv::Point(0,0))
    , _threshold(2)
    , _tracking(false)
  {
  } 

  void set_force_begin_action(void (*cb)(ForceTracker *sender)){ _force_begin_action = cb; }

  void set_force_move_action(void (*cb)(ForceTracker *sender)){ _force_move_action = cb; }

  void set_data(void *data){ _user_data = data; }

  cv::Point location()
  {
    return _current_point;
  }

  void* data(){ return _user_data; }

  virtual void send_point(cv::Point point)
  {
    std::cout << "ForceTracker(" << point.x << "," << point.y << ")" << std::endl;

    if(!_tracking){
      _tracking = true;
      _initial_point = point;
      _current_point = _initial_point;
      force_begin();
    } else{
      _current_point = point;
      force_move();
    }

  }
};


// class DragTracker : public InputTracker {
  
// };


// class DualForceTracker : public InputTracker {
// };


class InputTrackers {
private:
  std::vector<InputTracker*> _trackers;

public:
  void add(InputTracker *tracker){ _trackers.push_back(tracker); }

  void send_points(std::vector<cv::Point> points)
  {
    switch(points.size()){
    case 0:
      {
        std::cout << "point size: 0" << std::endl;
        break;
      }
    case 1:
      {
        std::cout << "point size: 1" << std::endl;
        std::vector<InputTracker*>::iterator tracker;
        for(tracker=_trackers.begin(); tracker!=_trackers.end(); ++tracker){
          (*tracker)->send_point(points.at(0));
        }
      }
      break;
    default:
      {
        std::cout << "point size: " << points.size() << std::endl;
        break;
      }
    }
  }
};


class HandDetector {
public:
  std::vector<cv::Point3i> detect(cv::Mat &image)
  {
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat thresholded;
    cv::Mat thresholded2;
    cv::Mat output;
    std::vector<std::vector<cv::Point> > contours;
    
    cv::cvtColor(image, gray, CV_BGR2GRAY); 

    cv::blur(gray, blurred, cv::Size(10,10));
    
    cv::threshold(blurred, thresholded, 70.0f, 255, CV_THRESH_BINARY);
    cv::threshold(blurred, thresholded2, 70.0f, 255, CV_THRESH_BINARY);

    cv::findContours(thresholded, contours, 
                     CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::cvtColor(thresholded2, output, CV_GRAY2BGR);  
    cv::namedWindow("hand");
    cv::imshow("hand", output);

    std::vector<cv::Point3i> points;
    for(std::vector<std::vector<cv::Point> >::iterator it=contours.begin(); it!=contours.end(); ++it){
      cv::Point2f center;
      float radius;
      std::vector<cv::Point> point = *it;
      cv::Mat pointsMatrix = cv::Mat(point);
      cv::minEnclosingCircle(pointsMatrix, center, radius);
      if(radius > 20.0f && radius < 200.0f){
        points.push_back(cv::Point3i(cv::saturate_cast<int>(center.x),
                                    cv::saturate_cast<int>(center.y),
                                    cv::saturate_cast<int>(radius)));
      }
    }
    return points;
  }

  void draw(cv::Mat &image, std::vector<cv::Point3i> points)
  {
    for(std::vector<cv::Point3i>::iterator point=points.begin(); point!=points.end(); ++point){
      cv::circle(image, cv::Point((*point).x, (*point).y), (*point).z, cv::Scalar(0,255,0));
      cv::circle(image, cv::Point((*point).x, (*point).y), 10, cv::Scalar(0, 0, 255));
    }
  }

  void detect_and_draw(cv::Mat &image)
  {
    std::vector<cv::Point3i> points;
    points = detect(image);
    draw(image, points);
  }

};


void force_begin(ForceTracker *tracker)
{
  cv::Point location = tracker->location();
  Window *window = reinterpret_cast<Window*>(tracker->data());
  std::cout << "force_begin(" << location.x << "," << location.y << ")" << std::endl;
  window->move(location);
}

void force_move(ForceTracker *tracker)
{
  cv::Point location = tracker->location();
  Window *window = reinterpret_cast<Window*>(tracker->data());
  std::cout << "force_move(" << location.x << "," << location.y << ")" << std::endl;
  window->move(location);
}



void run()
{
  Camera camera;
  Window window("Camera");
  FaceDetector face_detector;
  HandDetector hand_detector;
  InputTrackers trackers;
  ForceTracker *forcetracker = new ForceTracker();
  
  forcetracker->set_force_begin_action(force_begin);
  forcetracker->set_force_move_action(force_move);
  forcetracker->set_data(&window);
  trackers.add(forcetracker);

  window.show();
  while(1){
    if(window.wait_key())
      break;

    camera.capture();
    cv::Mat frame = camera.frame();

    // detect hands
    std::vector<cv::Point3i> hands = hand_detector.detect(frame);
    hand_detector.draw(frame, hands);

    std::vector<cv::Point> points;
    for(std::vector<cv::Point3i>::iterator it=hands.begin(); it!=hands.end(); ++it){
      points.push_back(cv::Point((*it).x, (*it).y));
    }
    trackers.send_points(points);

    // detect faces
    std::vector<cv::Rect> faces = face_detector.detect(frame);
    face_detector.draw(frame, faces);

    window.update_image(frame);

    if(faces.empty())
      continue;

    std::vector<cv::Rect>::const_iterator r;
    for(r=faces.begin(); r!=faces.end(); ++r){
      points.push_back(cv::Point(cv::saturate_cast<int>(r->x+r->width*0.5),
                                 cv::saturate_cast<int>(r->y+r->height*0.5)));
    }
    
    //    trackers.send_points(points);

  }

}

int main()
{
  run();
  return 0;
}
