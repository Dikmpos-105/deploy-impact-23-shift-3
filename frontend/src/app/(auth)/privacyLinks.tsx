import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";
import { useRouter } from "next/navigation";
import { Stack } from "@mui/material";

export default function LinksSection() {
  const router = useRouter();
  const handleClick = () => {
    router.replace("/privacyDeclaration");
  };
  return (
    <Box sx={{ fontSize: "12px", paddingBottom: 4 }}>
      <Typography sx={{ fontSize: "12px" }}>
        By clicking Sign In you agree to the SHIFT
      </Typography>
      <Stack flexDirection={"row"}>
        <Link
          component="button"
          onClick={handleClick}
          sx={{
            textDecoration: "underline",
            color: "#14366F",
          }}
        >
          User Agreement, Privacy Policy
        </Link>
        <Typography
          sx={{
            fontSize: "12px",
            display: "inline",
            marginX: 1,
          }}
        >
          and
        </Typography>
        <Link
          component="button"
          onClick={handleClick}
          sx={{
            textDecoration: "underline",
            color: "#14366F",
          }}
        >
          Cookie Policy
        </Link>
      </Stack>
    </Box>
  );
}
