"use client";
import * as React from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

import { redirect } from "next/navigation";

export default function HomePage() {
  // TODO: temp redirect until we have something to display here
  // redirect("/login");
  return (
    <Box sx={{ display: "flex", marginTop: "80px" }}>
      <div>
        <Typography variant="h1">
          Admin Dashboard -- - need content --
        </Typography>
      </div>
    </Box>
  );
}
