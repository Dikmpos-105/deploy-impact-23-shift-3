import { Roboto } from "next/font/google";
import { createTheme } from "@mui/material/styles";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  display: "swap",
});

const theme = createTheme({
  palette: {
    mode: "light",
    primary:{
      main:'#14366F'
    },
    secondary:{
      main:'#D7DDE7'
    }
  },

  typography: {
    fontFamily: roboto.style.fontFamily,
  },

  components: {

    MuiAlert: {
      styleOverrides: {
        root: ({ ownerState }) => ({
          ...(ownerState.severity === "info" && {
            backgroundColor: "#60a5fa",
          }),
        }),
      },
    },

    MuiLink: {
      styleOverrides: {
        "root": {
          "&.Mui-selected": {
            "backgroundColor": "pink"
          }
        }
 
      },
    },


    // MuiButtonBase:{
    //   defaultProps: {
    //     // The props to change the default for.
    //     disableRipple: true, // No more ripple, on the whole application 💣!
    //   },
    //   styleOverrides: {
    //     // Name of the slot
    //     root: {
    //       // Some CSS
    //       fontSize: '3rem',
    //     },
    //   }
      
    
    // }

    MuiButton: {
      variants: [
        {
          props: { variant: 'contained', },
          style: {
            borderRadius: '40px',
            //backgroundColor:'#14366F',
            // border: `2px dashed }`,
          },
          
        },
        {
          props: { variant: 'outlined', },
          style: {
            borderRadius: '40px',
            //backgroundColor:'#14366F',
            // border: `2px dashed }`,
          },
          
        },


        {
          props: { variant: 'contained', color: 'secondary' },
          style: {
            // border: `4px dashed `,
          },
        },
      ],
    },
  },
});

export default theme;
