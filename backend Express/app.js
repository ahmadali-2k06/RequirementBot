// 1. DOTENV MUST BE AT THE TOP
// This ensures variables exist before "authRoute" tries to use them.
require("dotenv").config(); 

//imports
const express = require("express");
const app = express();
const ejs = require("ejs");
const connectDB = require("./db/connect");
const errorHandler = require("./middlewares/errorHandler.js");

//api routes
const authRoute = require("./routes/api/auth");

//views routes
const loginRoute = require("./routes/views/login");

//security packages import
const helmet = require("helmet");
const { xss } = require("express-xss-sanitizer");
const cors = require("cors");
const cookieParser = require("cookie-parser");

//security middlewares
app.use(cookieParser());
app.set("trust proxy", 1);
app.use(helmet());
app.use(cors());
app.use(xss());

//middlewares
app.use(express.json());
app.use(express.static("./public"));
app.set("view engine", "ejs");

//static routes
app.use("/", loginRoute);
app.use("/login", loginRoute);

//api routes
// This allows your frontend fetch calls to work (e.g., /auth/register)
app.use("/auth", authRoute); 

//error Handler Middleware
app.use(errorHandler);

//app start
const PORT = process.env.PORT || 5000;
const start = async () => {
  try {
    await connectDB(process.env.MONGO_URI);
    app.listen(PORT, () => {
      console.log("App is listening on PORT " + PORT);
    });
  } catch (err) {
    console.log(err);
  }
};

start();