  const { StatusCodes } = require("http-status-codes");
  const jwt = require("jsonwebtoken");
  const nodemailer = require("nodemailer");
  const { BadRequestError, AuthenticationError } = require("../errors/errors");
  const User = require("../models/user");

  // Initialize email transporter (configure with your email service)
  const transporter = nodemailer.createTransport({
    service: process.env.EMAIL_SERVICE || "gmail",
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  });

  // Generate 4-digit OTP
  function generateOTP() {
    return Math.floor(1000 + Math.random() * 9000).toString();
  }

  // Send OTP Email
  async function sendOTPEmail(email, otp) {
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: "Your OTP for Precisely Registration",
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #2563eb;">Verify Your Email</h2>
          <p>Thank you for signing up with Precisely!</p>
          <p>Your 4-digit OTP code is:</p>
          <h1 style="color: #2563eb; letter-spacing: 5px; font-size: 32px;">${otp}</h1>
          <p>This code will expire in 10 minutes.</p>
          <p>Do not share this code with anyone.</p>
          <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 20px 0;">
          <p style="color: #6b7280; font-size: 12px;">If you didn't sign up for Precisely, please ignore this email.</p>
        </div>
      `,
    };

    return transporter.sendMail(mailOptions);
  }

  // Register user
  const registerUser = async (req, res) => {
    const { email, name, password, organization } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      throw new BadRequestError("Email already registered");
    }

    // Generate OTP
    const otp = generateOTP();
    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    // Create user with OTP (not verified yet)
    const user = await User.create({
      name,
      email,
      password,
      organization,
      otp,
      otpExpiry,
      isVerified: false,
    });

    // Send OTP email
    try {
      await sendOTPEmail(email, otp);
    } catch (error) {
      console.error("Email sending error:", error);
      // Delete user if email fails
      await User.findByIdAndDelete(user._id);
      throw new BadRequestError("Failed to send OTP. Please try again.");
    }

    res.status(StatusCodes.CREATED).json({
      msg: "OTP sent to your email. Please verify to complete registration.",
      email: email,
    });
  };

  // Verify OTP
  const verifyOTP = async (req, res) => {
    const { email, otp } = req.body;

    if (!email || !otp) {
      throw new BadRequestError("Email and OTP are required");
    }

    const user = await User.findOne({ email });
    if (!user) {
      throw new AuthenticationError("User not found");
    }

    // Check if already verified
    if (user.isVerified) {
      throw new BadRequestError("User already verified");
    }

    // Check OTP validity
    if (user.otp !== otp) {
      throw new AuthenticationError("Invalid OTP");
    }

    if (new Date() > user.otpExpiry) {
      throw new AuthenticationError("OTP expired. Please request a new one.");
    }

    // Mark user as verified
    user.isVerified = true;
    user.otp = null;
    user.otpExpiry = null;
    await user.save();

    // Create tokens
    const refreshToken = user.createRefreshJWT();
    const accessToken = user.createAccessJWT();
    user.refreshTokens.push(refreshToken);
    await user.save();

    res
      .status(StatusCodes.OK)
      .cookie("refreshToken", refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 7 * 24 * 60 * 60 * 1000,
      })
      .json({
        msg: "Email verified successfully!",
        user: { userID: user._id, name: user.name },
        accessToken,
      });
  };

  // Resend OTP
  const resendOTP = async (req, res) => {
    const { email } = req.body;

    if (!email) {
      throw new BadRequestError("Email is required");
    }

    const user = await User.findOne({ email });
    if (!user) {
      throw new AuthenticationError("User not found");
    }

    if (user.isVerified) {
      throw new BadRequestError("User already verified");
    }

    // Generate new OTP
    const otp = generateOTP();
    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000);

    user.otp = otp;
    user.otpExpiry = otpExpiry;
    await user.save();

    // Send OTP email
    try {
      await sendOTPEmail(email, otp);
    } catch (error) {
      console.error("Email sending error:", error);
      throw new BadRequestError("Failed to send OTP. Please try again.");
    }

    res.status(StatusCodes.OK).json({
      msg: "OTP resent successfully!",
    });
  };

  // Login user
  const loginUser = async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password) {
      throw new BadRequestError("Please provide all the credentials");
    }

    const user = await User.findOne({ email: email });
    if (!user) {
      throw new AuthenticationError("Invalid Credentials! Please try again.");
    }

    if (!user.isVerified) {
      throw new AuthenticationError("Please verify your email first");
    }

    const passwordCheckPassed = await user.checkPassword(password);
    if (!passwordCheckPassed) {
      throw new AuthenticationError("Invalid Credentials! Please try again.");
    }

    const isDemo = user.email === "demo@test.com";

    const refreshToken = user.createRefreshJWT();
    const accessToken = user.createAccessJWT({
      readonly: isDemo,
    });
    if (!isDemo) {
      await User.updateOne(
        { _id: user._id },
        { $push: { refreshTokens: refreshToken } }
      );
    }

    res
      .status(StatusCodes.OK)
      .cookie("refreshToken", refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 7 * 24 * 60 * 60 * 1000,
      })
      .json({
        user: {
          userID: user._id,
          name: user.name,
          readonly: isDemo,
        },
        accessToken,
      });
  };

  // Refresh access token
  const refreshAccessToken = async (req, res) => {
    const refreshToken = req.cookies.refreshToken;
    if (!refreshToken) {
      throw new BadRequestError("No refresh token provided");
    }
    try {
      const decoded = jwt.verify(refreshToken, process.env.JWT_SECRET_REFRESH);
      const user = await User.findById(decoded.userId);
      if (!user) {
        throw new AuthenticationError("Invalid token provided");
      }
      const isDemo = user.email === "demo@test.com";
      if (!isDemo && !user.refreshTokens.includes(refreshToken)) {
        throw new AuthenticationError("Invalid token provided");
      }
      const accessToken = user.createAccessJWT({ readonly: isDemo });
      return res.status(StatusCodes.OK).json({ accessToken });
    } catch (err) {
      const user = await User.findOne({ refreshTokens: refreshToken });
      if (user && user.email !== "demo@test.com") {
        user.refreshTokens = user.refreshTokens.filter((t) => t !== refreshToken);
        await user.save();
      }
      throw new AuthenticationError("Refresh token expired or invalid");
    }
  };

  // Demo login
  const demoLogin = async (req, res) => {
    const demoUser = await User.findOne({ email: "demo@test.com" });
    if (!demoUser) {
      throw new AuthenticationError("Demo user account not found.");
    }

    const refreshToken = demoUser.createRefreshJWT();
    const accessToken = demoUser.createAccessJWT({ readonly: true });

    res
      .status(StatusCodes.OK)
      .cookie("refreshToken", refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "strict",
        maxAge: 7 * 24 * 60 * 60 * 1000,
      })
      .json({
        user: {
          userID: demoUser._id,
          name: demoUser.name,
          readonly: true,
        },
        accessToken,
      });
  };

  // Logout
  const logOut = async (req, res) => {
    try {
      const refreshToken = req.cookies.refreshToken;
      if (!refreshToken) return res.sendStatus(StatusCodes.NO_CONTENT);

      const user = await User.findOne({ refreshTokens: refreshToken });
      if (!user) {
        res.clearCookie("refreshToken", {
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          sameSite: "strict",
        });
        return res.sendStatus(204);
      }

      user.refreshTokens = user.refreshTokens.filter(
        (token) => token !== refreshToken
      );
      await user.save();

      res.clearCookie("refreshToken", { httpOnly: true, sameSite: "strict" });

      return res
        .status(StatusCodes.OK)
        .json({ message: "Logged out successfully" });
    } catch (err) {
      console.error(err);
      res
        .status(StatusCodes.INTERNAL_SERVER_ERROR)
        .json({ message: "Server error during logout" });
    }
  };

  module.exports = {
    registerUser,
    verifyOTP,
    resendOTP,
    loginUser,
    refreshAccessToken,
    logOut,
    demoLogin,
  };