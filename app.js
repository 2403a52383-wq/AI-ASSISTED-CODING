const express = require('express');
const mongoose = require('mongoose');
const app = express();
app.use(express.json());

mongoose.connect('mongodb://localhost:27017/courseDB', { useNewUrlParser: true, useUnifiedTopology: true });

// Schemas
const userSchema = new mongoose.Schema({
    name: String,
    role: String, // 'student','faculty','admin'
    email: String,
    password: String,
    department: String
});
const User = mongoose.model('User', userSchema);

const courseSchema = new mongoose.Schema({
    courseCode: String,
    title: String,
    description: String,
    facultyId: String,
    totalSeats: Number,
    availableSeats: Number
});
const Course = mongoose.model('Course', courseSchema);

const enrollmentSchema = new mongoose.Schema({
    studentId: String,
    courseId: String,
    status: String, // 'Active', 'Dropped'
    enrolledAt: Date
});
const Enrollment = mongoose.model('Enrollment', enrollmentSchema);

// Authentication: Register & Login
app.post('/api/auth/register', async (req, res) => {
    const user = new User(req.body);
    await user.save();
    res.json({ message: 'Registered', user });
});

app.post('/api/auth/login', async (req, res) => {
    const user = await User.findOne({ email: req.body.email, password: req.body.password });
    if (!user) return res.status(401).json({ message: 'Invalid credentials' });
    res.json({ message: 'Logged in', user });
});

// Admin: Add Course
app.post('/api/courses', async (req, res) => {
    const course = new Course(req.body);
    course.availableSeats = course.totalSeats;
    await course.save();
    res.json({ message: 'Course Added', course });
});

// All: Get Courses
app.get('/api/courses', async (req, res) => {
    const courses = await Course.find();
    res.json(courses);
});

// Admin: Assign Faculty to Course
app.put('/api/courses/:id/assign', async (req, res) => {
    const course = await Course.findByIdAndUpdate(req.params.id, { facultyId: req.body.facultyId }, { new: true });
    res.json(course);
});

// Student: Enroll in Course
app.post('/api/enrollments', async (req, res) => {
    const course = await Course.findById(req.body.courseId);
    if (course.availableSeats < 1)
        return res.status(400).json({ message: 'No seats available' });
    const enrollment = new Enrollment({
        studentId: req.body.studentId,
        courseId: req.body.courseId,
        status: 'Active',
        enrolledAt: new Date()
    });
    await enrollment.save();
    course.availableSeats -= 1;
    await course.save();
    res.json({ message: 'Enrolled', enrollment });
});

// Student: Drop Course
app.delete('/api/enrollments/:id', async (req, res) => {
    const enrollment = await Enrollment.findByIdAndDelete(req.params.id);
    if (!enrollment) return res.status(404).json({ message: 'Enrollment not found' });
    const course = await Course.findById(enrollment.courseId);
    if (course) {
        course.availableSeats += 1;
        await course.save();
    }
    res.json({ message: 'Enrollment dropped' });
});

// Reports
app.get('/api/reports/enrollments', async (req, res) => {
    const enrollments = await Enrollment.aggregate([
        { $group: { _id: "$courseId", total: { $sum: 1 } } }
    ]);
    res.json(enrollments);
});

app.listen(3000, () => console.log('Server running on http://localhost:3000'));

/*
Use Postman or frontend to test these routes:
Register: POST /api/auth/register
Login: POST /api/auth/login
Add Course: POST /api/courses
Get Courses: GET /api/courses
Assign Faculty: PUT /api/courses/:id/assign
Enroll in Course: POST /api/enrollments
Drop Course: DELETE /api/enrollments/:id
Get Enrollment Report: GET /api/reports/enrollments
*/
