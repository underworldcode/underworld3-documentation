cl_1 = .04;
cl_2 = 0.005;
cl_2a = 0.003;
cl_3 = .02;
cl_4 = 0.01;
Point(1) = {-2, -1, 0, cl_1};
Point(2) = {-2, -1, 0, cl_1};
Point(3) = {2, -1, 0, cl_1};
Point(4) = {2, -0.75, 0, cl_1};
Point(5) = {2, 0, 0, cl_1};
Point(6) = {-2, 0, 0, cl_1};
Point(7) = {-2, -0.75, 0, cl_1};
Point(8) = {-0.08333333333329999, -0.75, 0, cl_2};
Point(9) = {0.08333333333329999, -0.75, 0, cl_2};
Point(10) = {0.08333333333329999, -0.6666666666667, 0, cl_2};
Point(11) = {-0.08333333333329999, -0.6666666666667, 0, cl_2};
Point(25) = {-.75, 0, 0, cl_4};
Point(26) = {.75, 0, 0, cl_4};
Point(27) = {0, 0, 0, cl_3};
Line(1) = {1, 3};
Line(2) = {3, 4};
Line(3) = {4, 5};
Line(4) = {5, 26};
Line(8) = {26, 27};
Line(9) = {27, 25};
Line(10) = {25, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};
Physical Line("Bottom",3) = {1};
Physical Line("Right",2) = {2, 3};
Physical Line("Left",1) = {7, 6};
Physical Line("Top",4) = {4, 8, 9, 10};
Point(12) = {-0.1033333333333, -0.75, 0, cl_2a};
Point(13) = {-0.0833333333333, -0.73, 0, cl_2a};
Point(14) = {-0.0833333333333, -0.686666666666666, 0, cl_2a};
Point(15) = {-0.0633333333333, -0.666666666666666, 0, cl_2a};
Point(16) = {0.0633333333333, -0.666666666666666, 0, cl_2a};
Point(17) = {0.0833333333333, -0.686666666666666, 0, cl_2a};
Point(18) = {0.0833333333333, -0.73, 0, cl_2a};
Point(19) = {0.1033333333333, -0.75, 0, cl_2a};
Point(20) = {-0.103333333333333, -0.73, 0, cl_2a};
Point(21) = {-0.063333333333333, -0.686666666666666, 0, cl_2a};
Point(22) = {0.063333333333333, -0.686666666666666, 0, cl_2a};
Point(24) = {0.103333333333333, -0.73, 0, cl_2a};
Circle(22) = {12, 20, 13};
Circle(23) = {14, 21, 15};
Circle(24) = {16, 22, 17};
Circle(25) = {18, 24, 19};
Line(26) = {7, 12};
Line(27) = {13, 14};
Line(28) = {15, 16};
Line(29) = {17, 18};
Line(30) = {19, 4};
Physical Line("LayerBoundary",5) = {26, 22, 27, 23, 28, 24, 29, 25, 30};
Line Loop(31) = {1, 2, -30, -25, -29, -24, -28, -23, -27, -22, -26, 7};
Plane Surface(32) = {31};
Line Loop(33) = {6, 26, 22, 27, 23, 28, 24, 29, 25, 30, 3, 4, 8, 9, 10};
Plane Surface(34) = {33};
Physical Surface("Weak", 10) = {32};
Physical Surface("Strong", 11) = {34};