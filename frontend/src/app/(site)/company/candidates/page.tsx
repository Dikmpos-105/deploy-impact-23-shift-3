"use client";
import * as React from "react";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import { SearchBar } from "@/app/(site)/company/searchBar";
import getModifiedListOfCandidates from "@/app/(site)/company/candidates/fetchMatchedCandidates";
import { CandidateList } from "@/app/(site)/company/candidates/candidateList";
import { useMemo, useState } from "react";
import Fuse from "fuse.js";
import { CandidateForJobListSingleMatch } from "@/app/(site)/company/candidates/types";
import IFuseOptions = Fuse.IFuseOptions;

const options: IFuseOptions<CandidateForJobListSingleMatch> = {
  includeScore: true,
  shouldSort: true,
  findAllMatches: true,
  threshold: 0,
  keys: ["job_title", "matching_score", "soft_skills", "hard_skills"],
};

export default function CandidatesPage() {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const listOfCandidates = getModifiedListOfCandidates();

  const filteredCandidates = useMemo(() => {
    const fuse = new Fuse(listOfCandidates, options);
    return fuse.search(searchTerm, { limit: 20 });
  }, [listOfCandidates, searchTerm]);

  return (
    <Container>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          margin: "-20px 0 0 0",
        }}
      >
        {listOfCandidates.length > 0 && (
          <SearchBar
            searchTerm={searchTerm}
            onSearch={setSearchTerm}
            placeholder={"Search for candidates"}
          />
        )}
        <CandidateList
          candidates={
            searchTerm
              ? filteredCandidates.map((fuse) => fuse.item)
              : listOfCandidates
          }
        />
      </Box>
    </Container>
  );
}